"""Load labels and construct chronological learner sequences."""

import csv
import json
from collections import defaultdict

from ..settings import CONCEPTS, DATA_DIR, OUTPUT_DIR, PROBE_QUESTIONS


def _key(row):
    return row["learner_id"], row["session_id"], row["task_id"]


def validate_evidence_coverage(rows, expected_rows):
    # Design item: Classification coverage
    # Current setting: Require one extracted label for every one of the 9,600 gold tasks.
    actual_keys = [_key(row) for row in rows]
    expected = {_key(row) for row in expected_rows}
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("complete evidence required: duplicate tasks found")
    actual = set(actual_keys)
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise ValueError(f"complete evidence required: missing={len(missing)}, extra={len(extra)}")


def load_course_maps():
    with (DATA_DIR / "questions.csv").open(encoding="utf-8") as handle:
        questions = list(csv.DictReader(handle))
    question_map = {
        question_id: index
        for index, question_id in enumerate(sorted(row["id"] for row in questions), 1)
    }
    concept_map = {concept: index for index, concept in enumerate(CONCEPTS)}
    question_concepts = {row["id"]: row["concept"] for row in questions}
    for concept, question_id in PROBE_QUESTIONS.items():
        # Confirm that every fixed test question belongs to its declared concept.
        if question_concepts.get(question_id) != concept:
            raise ValueError(f"fixed question {question_id} does not belong to {concept}")
    return question_map, concept_map


def load_splits():
    with (DATA_DIR / "splits.csv").open(encoding="utf-8") as handle:
        return {row["learner_id"]: row["split"] for row in csv.DictReader(handle)}


def load_records(label_source):
    # Design item: gold-SimpleKT diagnostic input
    # Current setting: Replace only LLM correctness labels with simulator labels.
    with (DATA_DIR / "gold_interactions.csv").open(encoding="utf-8") as handle:
        gold = list(csv.DictReader(handle))
    if label_source == "gold":
        return gold
    if label_source != "llm":
        raise ValueError("label_source must be 'llm' or 'gold'")
    rows = [
        json.loads(line)
        for line in (OUTPUT_DIR / "extracted_evidence.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    validate_evidence_coverage(rows, gold)
    return rows


def build_sequences(rows, split, question_map, concept_map):
    # Design item: SimpleKT input
    # Current setting: Ordered question, concept, and selected correctness-label sequences for each learner.
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["learner_id"]].append(row)

    sequences = []
    for learner_id in sorted(split):
        # Design item: Sequence order and length
        # Current setting: Sort by day, session, and task ID; every formal learner has 40 tasks.
        items = sorted(
            grouped[learner_id],
            key=lambda row: (int(row["day"]), row["session_id"], row["task_id"]),
        )
        sequences.append({
            "learner_id": learner_id,
            "split": split[learner_id],
            "question_ids": [question_map[row["question_id"]] for row in items],
            "concept_ids": [concept_map[row["concept_id"]] for row in items],
            "responses": [int(row["correct"]) for row in items],
            "days": [int(row["day"]) for row in items],
        })
    return sequences


def prepare_sequences(label_source):
    # Design item: gold-SimpleKT comparison control
    # Current setting: Keep model, hyperparameters, split, and training procedure identical to LLM-SimpleKT.
    question_map, concept_map = load_course_maps()
    sequences = build_sequences(
        load_records(label_source), load_splits(), question_map, concept_map
    )
    return sequences, question_map, concept_map
