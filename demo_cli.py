#!/usr/bin/env python3
"""
Demonstration CLI for Resume NER (Checkpoint 7).

Loads the production model (tagged PRD) from MLflow Model Registry
and runs inference on a sample resume text.

Usage:
    python demo_cli.py                                # run on sample text
    python demo_cli.py demo.input_file=path/to/file   # run on file
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import mlflow


SAMPLE_RESUME = """
John Smith
Senior Software Engineer - Google
San Francisco, CA - Email me: john.smith@example.com

WORK EXPERIENCE

Senior Software Engineer
Google -
January 2020 to Present
Role: Building distributed systems for search infrastructure.

Software Engineer
Microsoft -
June 2017 to December 2019

EDUCATION

M.S. in Computer Science
Stanford University - Stanford, CA
2015 to 2017

B.S. in Computer Engineering
MIT - Cambridge, MA
2011 to 2015

SKILLS
Python, Java, Go, Kubernetes, Docker, AWS, GCP, PostgreSQL, MongoDB, Machine Learning
"""


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=" * 70)
    print("RESUME NER — DEMO CLI (Checkpoint 7)")
    print("=" * 70)

    os.environ["MLFLOW_TRACKING_URI"] = cfg.mlflow.tracking_uri
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg.mlflow.s3_endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"] = cfg.mlflow.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg.mlflow.aws_secret_access_key

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Find the latest run tagged with PRD in the experiment
    print(f"\nLooking up latest run with tag stage={cfg.mlflow.prod_tag} ...")
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{cfg.mlflow.experiment_name}' not found. "
            f"Run train_cli.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.stage = '{cfg.mlflow.prod_tag}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No runs tagged '{cfg.mlflow.prod_tag}' found.")

    run = runs[0]
    print(f"  Found run: {run.info.run_id}")
    print(f"  Test Entity-F1: {run.data.metrics.get('test_entity_f1', '?'):.4f}")

    model_uri = f"runs:/{run.info.run_id}/model"
    print(f"\nLoading model from: {model_uri}")
    nlp = mlflow.spacy.load_model(model_uri)
    print(f"  Pipeline: {nlp.pipe_names}")
    print(f"  Labels: {nlp.get_pipe('ner').labels}")

    text = SAMPLE_RESUME
    print(f"\n{'=' * 70}")
    print("INPUT TEXT")
    print("=" * 70)
    print(text)

    doc = nlp(text)
    print(f"\n{'=' * 70}")
    print(f"PREDICTIONS ({len(doc.ents)} entities found)")
    print("=" * 70)
    for ent in doc.ents:
        print(f"  [{ent.label_:22s}] '{ent.text}'  ({ent.start_char}..{ent.end_char})")

    output = {
        "run_id": run.info.run_id,
        "test_entity_f1": run.data.metrics.get("test_entity_f1"),
        "text": text,
        "entities": [
            {"text": ent.text, "label": ent.label_,
             "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ],
    }
    out_path = Path("demo_output.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResult saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
