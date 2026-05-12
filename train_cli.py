#!/usr/bin/env python3
"""
Training CLI for Resume NER (Checkpoint 7).

Reproducible training of the best model with full MLflow logging.
Configuration via Hydra (.yaml).

Usage:
    python train_cli.py
    python train_cli.py training.n_epochs=50 model.type=baseline
    python train_cli.py data.use_augmentation=false
"""

from __future__ import annotations

import os
import random
import time
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import mlflow
import mlflow.spacy

from data_utils import (
    full_clean_pipeline,
    train_test_split,
    build_augmented_dataset,
    filter_overlapping_entities,
    get_entity_ruler_patterns,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_entity_metrics(nlp_model, test_data):
    """Entity-level strict-match metrics (overall + per-type)."""
    tp, fp, fn = 0, 0, 0
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for text, ann in test_data:
        true_set = {(s, e, l) for s, e, l in ann.get("entities", [])}
        pred_doc = nlp_model(text)
        pred_set = {(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents}

        matched = true_set & pred_set
        tp += len(matched)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

        for s, e, l in matched:
            per_type[l]["tp"] += 1
        for s, e, l in pred_set - true_set:
            per_type[l]["fp"] += 1
        for s, e, l in true_set - pred_set:
            per_type[l]["fn"] += 1

    def _prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        r = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    p, r, f1 = _prf(tp, fp, fn)
    per_type_metrics = {}
    for label in sorted(per_type):
        d = per_type[label]
        pp, rr, ff = _prf(d["tp"], d["fp"], d["fn"])
        per_type_metrics[label] = {
            "precision": pp, "recall": rr, "f1": ff,
            "support": d["tp"] + d["fn"],
        }
    return {
        "precision": p, "recall": r, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "per_type": per_type_metrics,
    }


def calculate_token_metrics(nlp_model, test_data):
    all_true, all_pred = [], []
    for text, ann in test_data:
        doc = nlp_model.make_doc(text)
        true_labels = ["O"] * len(doc)
        for start, end, label in ann.get("entities", []):
            for t in doc:
                if t.idx >= start and t.idx + len(t.text) <= end:
                    true_labels[t.i] = label
        pred_doc = nlp_model(text)
        pred_labels = [t.ent_type_ if t.ent_type_ else "O" for t in pred_doc]
        n = min(len(true_labels), len(pred_labels))
        all_true.extend(true_labels[:n])
        all_pred.extend(pred_labels[:n])
    acc = accuracy_score(all_true, all_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_type: str, base_spacy_model: str) -> spacy.Language:
    if model_type == "baseline":
        nlp = spacy.blank("en")
        nlp.add_pipe("ner", last=True)
        return nlp

    exclude = ["ner", "parser", "lemmatizer", "attribute_ruler",
               "tagger", "senter", "sentencizer"]
    try:
        nlp = spacy.load(base_spacy_model, exclude=exclude)
    except OSError:
        spacy.cli.download(base_spacy_model)
        nlp = spacy.load(base_spacy_model, exclude=exclude)

    for pipe_name in list(nlp.pipe_names):
        if pipe_name != "tok2vec":
            nlp.remove_pipe(pipe_name)

    if model_type == "hybrid":
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns(get_entity_ruler_patterns())

    nlp.add_pipe("ner", last=True)
    return nlp


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_spacy(nlp_model, train_data, dev_data, cfg):
    ner = nlp_model.get_pipe("ner")
    for _, ann in train_data:
        for _, _, lbl in ann.get("entities", []):
            ner.add_label(lbl)

    examples = []
    for text, ann in train_data[:10]:
        doc = nlp_model.make_doc(text)
        examples.append(Example.from_dict(doc, ann))

    keep = {"ner", "entity_ruler", "transformer"}
    other = [p for p in nlp_model.pipe_names if p not in keep]
    history = []

    with nlp_model.disable_pipes(*other):
        has_pretrained = "tok2vec" in nlp_model.pipe_names
        if has_pretrained:
            optimizer = nlp_model.resume_training(get_examples=lambda: examples)
        else:
            optimizer = nlp_model.initialize(get_examples=lambda: examples)

        best_f1 = 0.0
        best_weights = None
        no_improve = 0
        start = time.time()

        for epoch in range(1, cfg.training.n_epochs + 1):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(
                train_data,
                size=compounding(
                    cfg.training.batch_size_start,
                    cfg.training.batch_size_stop,
                    cfg.training.batch_size_compound,
                ),
            )
            for batch in batches:
                exs = [Example.from_dict(nlp_model.make_doc(t), a) for t, a in batch]
                nlp_model.update(exs, drop=cfg.training.dropout,
                                 sgd=optimizer, losses=losses)

            dev_m = calculate_entity_metrics(nlp_model, dev_data)
            dev_f1 = dev_m["f1"]
            loss = float(losses.get("ner", 0))

            history.append({
                "epoch": epoch, "loss": loss,
                "dev_f1": dev_f1,
                "dev_precision": dev_m["precision"],
                "dev_recall": dev_m["recall"],
            })
            print(f"  Epoch {epoch:2d}/{cfg.training.n_epochs}  "
                  f"loss={loss:.0f}  dev_F1={dev_f1:.4f}")

            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("dev_f1", dev_f1, step=epoch)
            mlflow.log_metric("dev_precision", dev_m["precision"], step=epoch)
            mlflow.log_metric("dev_recall", dev_m["recall"], step=epoch)

            if dev_f1 > best_f1:
                best_f1 = dev_f1
                best_weights = nlp_model.to_bytes()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.training.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        elapsed = time.time() - start

    if best_weights:
        nlp_model.from_bytes(best_weights)

    return nlp_model, history, elapsed, best_f1


# ---------------------------------------------------------------------------
# Artifacts (plots, examples)
# ---------------------------------------------------------------------------

def plot_learning_curve(history, out_path: Path):
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="steelblue")
    ax1.plot(df["epoch"], df["loss"], "o-", color="steelblue", label="loss")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Dev Entity-F1", color="orangered")
    ax2.plot(df["epoch"], df["dev_f1"], "s-", color="orangered", label="dev F1")
    ax2.tick_params(axis="y", labelcolor="orangered")
    plt.title("Learning curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(nlp_model, test_data, out_path: Path):
    """Build entity-level confusion-style matrix (true vs predicted labels)."""
    labels = sorted(set(
        l for _, ann in test_data for _, _, l in ann.get("entities", [])
    ))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n + 1, n + 1), dtype=int)
    extended = labels + ["<MISSED>"]

    for text, ann in test_data:
        true_set = {(s, e, l) for s, e, l in ann.get("entities", [])}
        pred_doc = nlp_model(text)
        pred_set = {(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents}

        true_by_span = {(s, e): l for s, e, l in true_set}
        pred_by_span = {(s, e): l for s, e, l in pred_set}

        for span, t_lbl in true_by_span.items():
            p_lbl = pred_by_span.get(span)
            ti = label_to_idx[t_lbl]
            if p_lbl is None:
                matrix[ti, n] += 1  # missed
            elif p_lbl in label_to_idx:
                matrix[ti, label_to_idx[p_lbl]] += 1

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=extended, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Entity-level confusion matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def save_prediction_examples(nlp_model, test_data, out_path: Path, n: int = 5):
    samples = []
    for text, ann in test_data[:n]:
        true = [(s, e, l, text[s:e]) for s, e, l in ann.get("entities", [])]
        pred_doc = nlp_model(text)
        pred = [(e.start_char, e.end_char, e.label_, e.text) for e in pred_doc.ents]
        samples.append({
            "text_preview": text[:300],
            "text_length": len(text),
            "true_entities": [{"start": s, "end": e, "label": l, "text": t}
                              for s, e, l, t in true],
            "predicted_entities": [{"start": s, "end": e, "label": l, "text": t}
                                   for s, e, l, t in pred],
        })
    out_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main (Hydra entrypoint)
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=" * 70)
    print("RESUME NER — TRAIN CLI (Checkpoint 7)")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))

    # Set seeds for reproducibility
    random.seed(cfg.data.random_state)
    np.random.seed(cfg.data.random_state)

    # Configure MLflow env
    os.environ["MLFLOW_TRACKING_URI"] = cfg.mlflow.tracking_uri
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg.mlflow.s3_endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"] = cfg.mlflow.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg.mlflow.aws_secret_access_key

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Resolve dataset path relative to the project root (hydra changes cwd)
    project_root = Path(hydra.utils.get_original_cwd())
    dataset_path = project_root / cfg.data.dataset_path

    print(f"\nLoading dataset: {dataset_path}")
    data = full_clean_pipeline(dataset_path)

    train_data, test_data = train_test_split(
        data, test_size=cfg.data.test_size, random_state=cfg.data.random_state)
    train_data, dev_data = train_test_split(
        train_data, test_size=cfg.data.dev_size,
        random_state=cfg.data.random_state + 1)

    print(f"\nSplits: train={len(train_data)} dev={len(dev_data)} test={len(test_data)}")

    if cfg.data.use_augmentation:
        train_data_used = build_augmented_dataset(train_data,
                                                   random_state=cfg.data.random_state)
        train_data_used = filter_overlapping_entities(
            train_data_used, strategy="keep_longer")
    else:
        train_data_used = train_data

    print(f"Train set used: {len(train_data_used)} resumes")

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        # Log config
        mlflow.log_params({
            "model_type": cfg.model.type,
            "base_spacy_model": cfg.model.base_spacy_model,
            "n_epochs": cfg.training.n_epochs,
            "dropout": cfg.training.dropout,
            "patience": cfg.training.patience,
            "use_augmentation": cfg.data.use_augmentation,
            "random_state": cfg.data.random_state,
            "test_size": cfg.data.test_size,
            "dev_size": cfg.data.dev_size,
            "n_train": len(train_data),
            "n_train_augmented": len(train_data_used),
            "n_dev": len(dev_data),
            "n_test": len(test_data),
        })

        # Log config artifact
        with tempfile.TemporaryDirectory() as tmpd:
            tmpd_p = Path(tmpd)
            cfg_path = tmpd_p / "config.yaml"
            cfg_path.write_text(OmegaConf.to_yaml(cfg))
            mlflow.log_artifact(str(cfg_path))

            # Build & train
            print(f"\nBuilding model: {cfg.model.type}")
            nlp = build_model(cfg.model.type, cfg.model.base_spacy_model)

            print(f"Training...")
            nlp, history, train_time, best_dev_f1 = train_spacy(
                nlp, train_data_used, dev_data, cfg)

            # Final test metrics
            print(f"\nEvaluating on test set...")
            test_ent = calculate_entity_metrics(nlp, test_data)
            test_tok = calculate_token_metrics(nlp, test_data)

            print(f"\nTest Entity-F1 = {test_ent['f1']:.4f}")
            print(f"Test Entity-P  = {test_ent['precision']:.4f}")
            print(f"Test Entity-R  = {test_ent['recall']:.4f}")
            print(f"Test Token-Acc = {test_tok['accuracy']:.4f}")

            mlflow.log_metrics({
                "test_entity_f1": test_ent["f1"],
                "test_entity_precision": test_ent["precision"],
                "test_entity_recall": test_ent["recall"],
                "test_token_accuracy": test_tok["accuracy"],
                "test_token_f1": test_tok["f1"],
                "best_dev_f1": best_dev_f1,
                "train_time_sec": train_time,
            })

            # Per-entity metrics
            for label, m in test_ent["per_type"].items():
                safe_label = label.replace(" ", "_").lower()
                mlflow.log_metric(f"test_f1_{safe_label}", m["f1"])
                mlflow.log_metric(f"test_precision_{safe_label}", m["precision"])
                mlflow.log_metric(f"test_recall_{safe_label}", m["recall"])

            # Plots
            print("\nLogging artifacts...")
            plot_learning_curve(history, tmpd_p / "learning_curve.png")
            plot_confusion_matrix(nlp, test_data, tmpd_p / "confusion_matrix.png")
            save_prediction_examples(nlp, test_data, tmpd_p / "prediction_examples.json")

            mlflow.log_artifact(str(tmpd_p / "learning_curve.png"))
            mlflow.log_artifact(str(tmpd_p / "confusion_matrix.png"))
            mlflow.log_artifact(str(tmpd_p / "prediction_examples.json"))

            # Save model
            model_dir = tmpd_p / "model"
            nlp.to_disk(model_dir)
            mlflow.spacy.log_model(spacy_model=nlp, artifact_path="model",
                                    registered_model_name=cfg.mlflow.register_model_name)

            # Tag as production
            mlflow.set_tag("stage", cfg.mlflow.prod_tag)
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("augmentation", str(cfg.data.use_augmentation))

        print(f"\nMLflow run id:   {run.info.run_id}")
        print(f"MLflow run URL:  {cfg.mlflow.tracking_uri}/#/experiments/"
              f"{run.info.experiment_id}/runs/{run.info.run_id}")
        print(f"Tagged as:       {cfg.mlflow.prod_tag}")
        print(f"Registered as:   {cfg.mlflow.register_model_name}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
