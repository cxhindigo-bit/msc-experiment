"""Predict eight fixed-test-question probabilities directly from dialogue history."""

import csv
import json
import math

from ..llm_client import ask_json, runtime_manifest
from ..settings import (
    CHECKPOINTS,
    CONCEPT_NAMES,
    CONCEPTS,
    DATA_DIR,
    LLM_TEMPERATURE,
    MAX_LLM_ATTEMPTS,
    OUTPUT_DIR,
    PROBE_QUESTIONS,
)


SYSTEM = "Estimate fixed-question correctness probabilities from the visible tutoring history. Return JSON only."
SCHEMA = {
    "type": "object",
    "properties": {
        "probabilities": {
            "type": "object",
            "properties": {concept: {"type": "number"} for concept in CONCEPTS},
            "required": CONCEPTS,
            "additionalProperties": False,
        }
    },
    "required": ["probabilities"],
    "additionalProperties": False,
}


def validate_probabilities(values):
    # Design item: Pure-LLM output validation
    # Current setting: Require exactly F1–F8 with numeric values in [0, 1]; never fill invalid output with defaults.
    if not isinstance(values, dict):
        raise ValueError("probabilities must be an object")
    missing = [concept for concept in CONCEPTS if concept not in values]
    extra = [concept for concept in values if concept not in CONCEPTS]
    if missing:
        raise ValueError(f"missing concepts: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unexpected concepts: {', '.join(extra)}")
    result = {}
    for concept in CONCEPTS:
        value = values[concept]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{concept} must be numeric")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{concept} must be within [0, 1]")
        result[concept] = float(value)
    return result


def _read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prompt(learner_id, day, sessions, question_text):
    # Design item: Prediction unit
    # Current setting: One learner, one checkpoint, one concept, and that concept's fixed test question.
    tests = "\n".join(
        f"- {concept}: {CONCEPT_NAMES[concept]}; fixed question "
        f"{PROBE_QUESTIONS[concept]}: {question_text[PROBE_QUESTIONS[concept]]}"
        for concept in CONCEPTS
    )
    history = "\n".join(
        f"[day {session['day']} | {turn['question_id']}] "
        f"{turn['speaker'].upper()}: {turn['text']}"
        for session in sessions
        for turn in session["turns"]
    )
    return f"""Learner: {learner_id}
Prediction day: {day}

Estimate the probability that the learner answers each fixed question correctly
on the first attempt without a teacher hint. Use only dialogue dated on or before
the prediction day.

FIXED QUESTIONS
{tests}

VISIBLE DIALOGUE
{history}"""


def run():
    # Design item: Pure-LLM input
    # Current setting: Same dialogue source as LLM-SimpleKT, with all history before the checkpoint and F1–F8 descriptions.
    # Design item: Hidden simulator information
    # Current setting: Neither pipeline receives learning rates, forgetting rates, draw values, or reference probabilities.
    dialogue_path = DATA_DIR / "raw_dialogues.jsonl"
    sessions = [
        json.loads(line)
        for line in dialogue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Design item: Test cohort
    # Current setting: The same 48 test learners used by both main pipelines and gold-SimpleKT.
    test_ids = [
        row["learner_id"] for row in _read_csv(DATA_DIR / "splits.csv")
        if row["split"] == "test"
    ]
    question_text = {
        row["id"]: row["text"] for row in _read_csv(DATA_DIR / "questions.csv")
    }
    predictions, failures = [], []
    for learner_id in test_ids:
        for day in CHECKPOINTS:
            # Design item: Visible history
            # Current setting: Day 10 uses 20 tasks; Day 20 uses all 40 tasks from the shared dialogue dataset.
            visible = [
                session for session in sessions
                if session["learner_id"] == learner_id and int(session["day"]) <= day
            ]
            error = None
            for attempt in range(MAX_LLM_ATTEMPTS):
                try:
                    data = ask_json(
                        "pure_llm",
                        f"{learner_id}-day-{day}-attempt-{attempt}",
                        SYSTEM,
                        _prompt(learner_id, day, visible, question_text),
                        SCHEMA,
                        LLM_TEMPERATURE,
                    )
                    values = validate_probabilities(data["probabilities"])
                    predictions.extend(
                        {
                            "learner_id": learner_id,
                            "day": day,
                            "concept_id": concept,
                            "probability": round(values[concept], 8),
                        }
                        for concept in CONCEPTS
                    )
                    break
                except Exception as exc:
                    error = str(exc)
            else:
                failures.append({"learner_id": learner_id, "day": day, "error": error})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(
        OUTPUT_DIR / "pure_llm_predictions.csv",
        ["learner_id", "day", "concept_id", "probability"],
        predictions,
    )
    _write_csv(
        OUTPUT_DIR / "pure_llm_failures.csv",
        ["learner_id", "day", "error"],
        failures,
    )
    (OUTPUT_DIR / "pure_llm_manifest.json").write_text(
        json.dumps({**runtime_manifest(), "temperature": LLM_TEMPERATURE}, indent=2),
        encoding="utf-8",
    )
    return OUTPUT_DIR / "pure_llm_predictions.csv"


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
