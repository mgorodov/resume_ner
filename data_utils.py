"""
Data utilities for Resume NER: loading, cleaning, augmentation.
Fixes known issues with the Dataturks dataset and provides
data augmentation to increase effective training set size.
"""

import json
import re
import copy
import random
import math
from pathlib import Path
from collections import Counter, defaultdict

import spacy


# ---------------------------------------------------------------------------
# 1. Loading & parsing
# ---------------------------------------------------------------------------

def convert_dataturks_to_spacy(filepath: str | Path) -> list[tuple[str, dict]]:
    """
    Convert Dataturks JSON to SpaCy training format.
    Uses original annotation offsets with validation and fallback.
    """
    nlp = spacy.blank("en")
    training_data = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_idx, line in enumerate(lines):
        data = json.loads(line)
        raw_text = data["content"]
        text = raw_text.replace("\n", " ").replace("\t", " ").strip()
        doc = nlp.make_doc(text)

        entities = []
        annotations = data.get("annotation")
        if annotations is None:
            training_data.append((text, {"entities": []}))
            continue

        for annotation in annotations:
            point = annotation["points"][0]
            labels = annotation["label"]
            if not isinstance(labels, list):
                labels = [labels]

            for label in labels:
                point_text = point["text"].strip()
                if not point_text:
                    continue

                orig_start = point["start"]
                orig_end = point["end"]

                # Adjust offsets: the original JSON has offsets for the raw text
                # (with \n\t).  After replace(\n->" ", \t->" ") the char count
                # stays the same, so offsets still valid.  But end is sometimes
                # inclusive — detect and fix.
                if orig_end - orig_start == len(point_text) - 1:
                    orig_end += 1  # make exclusive

                candidate = text[orig_start:orig_end].strip()
                norm_point = _normalise(point_text)

                if _normalise(candidate) == norm_point:
                    match_start, match_end = orig_start, orig_end
                else:
                    # Fallback: regex search (first occurrence — same as old code)
                    matches = list(re.finditer(re.escape(point_text), text))
                    if not matches:
                        continue
                    match_start, match_end = matches[0].span()

                # Snap to token boundaries
                match_start, match_end = _snap_to_tokens(doc, match_start, match_end)
                if match_start is None:
                    continue

                entities.append((match_start, match_end, label))

        training_data.append((text, {"entities": entities}))

    return training_data


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _snap_to_tokens(doc, start: int, end: int):
    """Snap character offsets to the nearest token boundaries."""
    token_starts = {t.idx: t for t in doc}
    token_ends = {t.idx + len(t.text): t for t in doc}

    if start in token_starts and end in token_ends:
        return start, end

    best_start = None
    best_end = None
    for t in doc:
        t_start = t.idx
        t_end = t.idx + len(t.text)
        if t_start <= start < t_end:
            best_start = t_start
        if t_start < end <= t_end:
            best_end = t_end

    if best_start is not None and best_end is not None and best_start < best_end:
        return best_start, best_end
    return None, None


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------

def trim_entity_spans(data: list) -> list:
    """Remove leading/trailing whitespace from entity spans."""
    ws = re.compile(r"\s")
    cleaned = []
    for text, ann in data:
        entities = []
        for start, end, label in ann["entities"]:
            while start < end and start < len(text) and ws.match(text[start]):
                start += 1
            while end > start and ws.match(text[end - 1]):
                end -= 1
            if start < end:
                entities.append((start, end, label))
        cleaned.append((text, {"entities": entities}))
    return cleaned


def remove_unknown_label(data: list) -> list:
    """Drop entities with the UNKNOWN label — they add noise."""
    cleaned = []
    removed = 0
    for text, ann in data:
        ents = [(s, e, l) for s, e, l in ann["entities"] if l != "UNKNOWN"]
        removed += len(ann["entities"]) - len(ents)
        cleaned.append((text, {"entities": ents}))
    if removed:
        print(f"  [clean] Removed {removed} UNKNOWN entities")
    return cleaned


def filter_empty_resumes(data: list) -> list:
    """Remove resumes with zero entities — nothing to learn from."""
    before = len(data)
    data = [(t, a) for t, a in data if a["entities"]]
    after = len(data)
    if before != after:
        print(f"  [clean] Removed {before - after} resumes with no entities")
    return data


def filter_overlapping_entities(data: list, strategy: str = "keep_longer") -> list:
    """
    Handle overlapping entity spans.
    strategy='keep_longer': keep the longer span on overlap (preserves more info).
    strategy='keep_first': keep the first span (original behaviour).
    """
    cleaned = []
    total_overlaps = 0
    total_resolved = 0

    for text, ann in data:
        entities = list(ann.get("entities", []))
        if not entities:
            cleaned.append((text, ann))
            continue

        entities.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        non_overlapping = []
        for start, end, label in entities:
            if not non_overlapping:
                non_overlapping.append((start, end, label))
                continue

            prev_s, prev_e, prev_l = non_overlapping[-1]
            if start < prev_e:  # overlap
                total_overlaps += 1
                if strategy == "keep_longer":
                    cur_len = end - start
                    prev_len = prev_e - prev_s
                    if cur_len > prev_len:
                        non_overlapping[-1] = (start, end, label)
                        total_resolved += 1
                # else: keep_first — just skip current
            else:
                non_overlapping.append((start, end, label))

        cleaned.append((text, {"entities": non_overlapping}))

    if total_overlaps:
        print(f"  [clean] {total_overlaps} overlapping entities found, "
              f"{total_resolved} resolved by keeping longer span")
    return cleaned


def validate_entities(data: list) -> list:
    """Drop entities where text[start:end] is empty or purely whitespace."""
    cleaned = []
    dropped = 0
    for text, ann in data:
        valid = []
        for start, end, label in ann["entities"]:
            span_text = text[start:end].strip()
            if span_text and start < end <= len(text):
                valid.append((start, end, label))
            else:
                dropped += 1
        cleaned.append((text, {"entities": valid}))
    if dropped:
        print(f"  [clean] Dropped {dropped} invalid entity spans")
    return cleaned


def full_clean_pipeline(filepath: str | Path) -> list:
    """Run the complete cleaning pipeline end-to-end."""
    print("Loading and cleaning data...")
    data = convert_dataturks_to_spacy(filepath)
    data = trim_entity_spans(data)
    data = remove_unknown_label(data)
    data = validate_entities(data)
    data = filter_overlapping_entities(data, strategy="keep_longer")
    data = filter_empty_resumes(data)
    print(f"  Final: {len(data)} resumes\n")
    return data


# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------

def train_test_split(data: list, test_size: float = 0.1, random_state: int = 42):
    data = list(data)
    random.Random(random_state).shuffle(data)
    split = len(data) - math.floor(test_size * len(data))
    return data[:split], data[split:]


# ---------------------------------------------------------------------------
# 4. Data augmentation
# ---------------------------------------------------------------------------

_SKILL_SYNONYMS = {
    "Java": ["Kotlin", "Scala", "Groovy"],
    "Python": ["Ruby", "Perl"],
    "C++": ["C#", "Rust", "Go"],
    "JavaScript": ["TypeScript", "CoffeeScript"],
    "React": ["Vue.js", "Angular", "Svelte"],
    "AWS": ["Azure", "GCP", "IBM Cloud"],
    "Docker": ["Kubernetes", "Podman"],
    "MySQL": ["PostgreSQL", "MariaDB", "SQLite"],
    "MongoDB": ["CouchDB", "Redis", "DynamoDB"],
    "Linux": ["Unix", "FreeBSD"],
    "TensorFlow": ["PyTorch", "Keras", "MXNet"],
    "Machine Learning": ["Deep Learning", "Data Science", "AI"],
    "HTML": ["HTML5"],
    "CSS": ["CSS3", "SASS", "LESS"],
    "SQL": ["PL/SQL", "T-SQL", "NoSQL"],
    "Git": ["SVN", "Mercurial"],
    "Agile": ["Scrum", "Kanban"],
}

_INDIAN_NAMES = [
    "Rahul Sharma", "Priya Patel", "Amit Kumar", "Sneha Reddy",
    "Vikram Singh", "Ananya Gupta", "Rohit Verma", "Deepa Nair",
    "Arjun Mehta", "Kavita Iyer", "Suresh Rao", "Divya Joshi",
    "Ravi Shankar", "Meera Kapoor", "Anil Deshmukh", "Pooja Agarwal",
    "Sanjay Pillai", "Swati Mishra", "Karan Malhotra", "Nisha Bhat",
]

_INDIAN_CITIES = [
    "Bengaluru, Karnataka", "Mumbai, Maharashtra", "Hyderabad, Telangana",
    "Chennai, Tamil Nadu", "Pune, Maharashtra", "Delhi",
    "Kolkata, West Bengal", "Ahmedabad, Gujarat", "Noida, Uttar Pradesh",
    "Gurgaon, Haryana", "Jaipur, Rajasthan", "Kochi, Kerala",
    "Chandigarh", "Lucknow, Uttar Pradesh", "Coimbatore, Tamil Nadu",
]

_COMPANIES = [
    "Infosys", "TCS", "Wipro", "HCL Technologies", "Tech Mahindra",
    "Cognizant", "Accenture", "Capgemini", "IBM India", "Oracle India",
    "Microsoft India", "Google India", "Amazon", "Flipkart", "Paytm",
    "SAP Labs India", "Adobe India", "Cisco Systems", "Dell Technologies",
    "Deloitte",
]


def augment_entity_swap(data: list, n_augmented: int = 1,
                        random_state: int = 42) -> list:
    """
    Create augmented copies by swapping entity values (names, cities,
    companies) with random alternatives from pools.
    """
    rng = random.Random(random_state)
    augmented = []

    entity_pools = defaultdict(list)
    for text, ann in data:
        for start, end, label in ann["entities"]:
            val = text[start:end].strip()
            if val:
                entity_pools[label].append(val)

    for label in ("Name",):
        entity_pools[label].extend(_INDIAN_NAMES)
    for label in ("Location",):
        entity_pools[label].extend(_INDIAN_CITIES)
    for label in ("Companies worked at",):
        entity_pools[label].extend(_COMPANIES)

    swappable = {"Name", "Location", "Companies worked at"}

    for text, ann in data:
        for _ in range(n_augmented):
            new_text = text
            new_entities = []
            offset = 0

            sorted_ents = sorted(ann["entities"], key=lambda x: x[0])

            for start, end, label in sorted_ents:
                adj_start = start + offset
                adj_end = end + offset
                old_val = new_text[adj_start:adj_end]

                if label in swappable and entity_pools[label]:
                    new_val = rng.choice(entity_pools[label])
                    new_text = new_text[:adj_start] + new_val + new_text[adj_end:]
                    new_end = adj_start + len(new_val)
                    offset += len(new_val) - len(old_val)
                    new_entities.append((adj_start, new_end, label))
                else:
                    new_entities.append((adj_start, adj_end, label))

            augmented.append((new_text, {"entities": new_entities}))

    return augmented


def augment_case_variation(data: list, random_state: int = 42) -> list:
    """
    Create augmented copies with random case changes on entity text:
    UPPER, lower, or Title case.
    """
    rng = random.Random(random_state)
    augmented = []

    case_fns = [str.upper, str.lower, str.title]

    for text, ann in data:
        new_text = text
        new_entities = []
        offset = 0

        sorted_ents = sorted(ann["entities"], key=lambda x: x[0])
        for start, end, label in sorted_ents:
            adj_start = start + offset
            adj_end = end + offset
            old_val = new_text[adj_start:adj_end]

            fn = rng.choice(case_fns)
            new_val = fn(old_val)

            new_text = new_text[:adj_start] + new_val + new_text[adj_end:]
            new_end = adj_start + len(new_val)
            offset += len(new_val) - len(old_val)
            new_entities.append((adj_start, new_end, label))

        augmented.append((new_text, {"entities": new_entities}))

    return augmented


def augment_skill_synonym(data: list, prob: float = 0.3,
                          random_state: int = 42) -> list:
    """
    Replace skill mentions with synonyms from a predefined dictionary.
    Only affects entities labeled 'Skills'.
    """
    rng = random.Random(random_state)
    rev_map = {}
    for orig, syns in _SKILL_SYNONYMS.items():
        rev_map[orig.lower()] = syns

    augmented = []
    for text, ann in data:
        new_text = text
        new_entities = []
        offset = 0

        sorted_ents = sorted(ann["entities"], key=lambda x: x[0])
        for start, end, label in sorted_ents:
            adj_start = start + offset
            adj_end = end + offset
            old_val = new_text[adj_start:adj_end]

            if label == "Skills" and rng.random() < prob:
                old_lower = old_val.strip().lower()
                if old_lower in rev_map:
                    new_val = rng.choice(rev_map[old_lower])
                    new_text = new_text[:adj_start] + new_val + new_text[adj_end:]
                    new_end = adj_start + len(new_val)
                    offset += len(new_val) - len(old_val)
                    new_entities.append((adj_start, new_end, label))
                    continue

            new_entities.append((adj_start, adj_end, label))

        augmented.append((new_text, {"entities": new_entities}))

    return augmented


def build_augmented_dataset(train_data: list, random_state: int = 42) -> list:
    """Apply all augmentation strategies and combine with original data."""
    print("Augmenting training data...")
    aug1 = augment_entity_swap(train_data, n_augmented=1, random_state=random_state)
    aug2 = augment_case_variation(train_data, random_state=random_state + 1)
    aug3 = augment_skill_synonym(train_data, prob=0.3, random_state=random_state + 2)

    combined = train_data + aug1 + aug2 + aug3
    print(f"  Original: {len(train_data)}, "
          f"After augmentation: {len(combined)} "
          f"({len(combined)/len(train_data):.1f}x)\n")
    return combined


# ---------------------------------------------------------------------------
# 5. Entity Ruler patterns
# ---------------------------------------------------------------------------

def get_entity_ruler_patterns() -> list[dict]:
    """Return rule-based patterns for structured entities."""
    patterns = []

    patterns.append({"label": "Email Address",
                     "pattern": [{"TEXT": {"REGEX": r"[\w.+-]+@[\w-]+\.[\w.]+"}}]})

    for year in range(1980, 2031):
        patterns.append({"label": "Graduation Year",
                         "pattern": [{"TEXT": str(year)}]})

    exp_patterns = [
        [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years", "yr", "yrs"]}}],
        [{"LIKE_NUM": True}, {"TEXT": "+"}, {"LOWER": {"IN": ["year", "years", "yr", "yrs"]}}],
        [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years", "yr", "yrs"]}},
         {"LOWER": "of"}, {"LOWER": "experience"}],
        [{"LIKE_NUM": True}, {"TEXT": "+"}, {"LOWER": {"IN": ["year", "years", "yr", "yrs"]}},
         {"LOWER": "of"}, {"LOWER": "experience"}],
    ]
    for pat in exp_patterns:
        patterns.append({"label": "Years of Experience", "pattern": pat})

    return patterns


# ---------------------------------------------------------------------------
# 6. Statistics
# ---------------------------------------------------------------------------

def dataset_stats(data: list, name: str = "Dataset") -> dict:
    """Print and return entity statistics."""
    label_counts = Counter()
    total_entities = 0
    for _, ann in data:
        for _, _, label in ann["entities"]:
            label_counts[label] += 1
            total_entities += 1

    print(f"--- {name} ---")
    print(f"  Resumes: {len(data)}, Entities: {total_entities}")
    for label, count in label_counts.most_common():
        print(f"    {label}: {count}")
    print()
    return {"n_resumes": len(data), "n_entities": total_entities,
            "label_counts": dict(label_counts)}
