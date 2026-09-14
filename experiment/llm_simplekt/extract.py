"""Classify only the first student answer in each five-task session."""

import csv
import json

from ..llm_client import ask_json, runtime_manifest
from ..settings import DATA_DIR, LLM_TEMPERATURE, MAX_LLM_ATTEMPTS, OUTPUT_DIR


SYSTEM = "Judge whether each initial student answer is correct. Return JSON only."
ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "question_id": {"type": "string"},
        "correct": {"type": "integer", "enum": [0, 1]},
    },
    "required": ["task_id", "question_id", "correct"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array", "minItems": 5, "maxItems": 5, "items": ITEM_SCHEMA,
        }
    },
    "required": ["classifications"],
    "additionalProperties": False,
}


def _source_tasks(session):
    # Design item: Initial-answer location
    # Current setting: Use the first student utterance in each of the session's five tasks.
    tasks = []
    for turn in session["turns"]:
        if turn["speaker"] != "student":
            continue
        if any(row["task_id"] == turn["task_id"] for row in tasks):
            continue
        tasks.append({
            "task_id": turn["task_id"],
            "question_id": turn["question_id"],
            "source_turn": turn["turn_id"],
        })
    if len(tasks) != 5:
        raise ValueError(f"{session['session_id']} must contain five initial answers")
    return tasks


def validate_classification(returned, expected):
    # Require five aligned task IDs and binary correctness values.
    if not isinstance(returned, list) or len(returned) != len(expected):
        raise ValueError("classification must contain five tasks")
    for item, source in zip(returned, expected):
        if (item.get("task_id"), item.get("question_id")) != (
            source["task_id"], source["question_id"]
        ):
            raise ValueError("classification task order does not match the source session")
        if item.get("correct") not in (0, 1):
            raise ValueError("correct must be 0 or 1")
    return returned


def _prompt(session, source_tasks):
    # Design item: Initial-answer classification task
    # Current setting: Separate five tasks, locate each initial answer, judge correctness, and preserve task alignment.
    transcript = "\n".join(
        f"[{turn['turn_id']} | {turn['task_id']} | {turn['question_id']}] "
        f"{turn['speaker'].upper()}: {turn['text']}"
        for turn in session["turns"]
    )
    expected = ", ".join(
        f"{row['task_id']}:{row['question_id']}" for row in source_tasks
    )
    return f"""Return one correct/incorrect judgement for the first student answer
of each task. Ignore later corrected answers. Keep this task order: {expected}.

TRANSCRIPT
{transcript}"""


def run():
    # Design item: LLM-SimpleKT language input
    # Current setting: Read the same dialogue dataset as Pure-LLM, one five-task session per request.
    sessions = [
        json.loads(line)
        for line in (DATA_DIR / "raw_dialogues.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    concepts = {}
    with (DATA_DIR / "questions.csv").open(encoding="utf-8") as handle:
        # Use the authoritative question-to-concept map; the LLM does not infer concepts.
        concepts = {row["id"]: row["concept"] for row in csv.DictReader(handle)}

    evidence, failures = [], []
    # Classify every available session; write sessions that still fail after retries separately.
    for session in sessions:
        source_tasks = _source_tasks(session)
        error = None
        for attempt in range(MAX_LLM_ATTEMPTS):
            try:
                data = ask_json(
                    "initial_answer",
                    f"{session['session_id']}-attempt-{attempt}",
                    SYSTEM,
                    _prompt(session, source_tasks),
                    SCHEMA,
                    LLM_TEMPERATURE,
                )
                items = validate_classification(data["classifications"], source_tasks)
                # Design item: Extracted evidence
                # Current setting: LLM supplies correct=0/1; the program supplies learner, session, day, question, concept, and order.
                for item, source in zip(items, source_tasks):
                    evidence.append({
                        "learner_id": session["learner_id"],
                        "session_id": session["session_id"],
                        "task_id": source["task_id"],
                        "day": int(session["day"]),
                        "source_turn": source["source_turn"],
                        "question_id": source["question_id"],
                        "concept_id": concepts[source["question_id"]],
                        "correct": item["correct"],
                    })
                break
            except Exception as exc:
                error = str(exc)
        else:
            failures.append({"session_id": session["session_id"], "error": error})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence_path = OUTPUT_DIR / "extracted_evidence.jsonl"
    evidence_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evidence),
        encoding="utf-8",
    )
    with (OUTPUT_DIR / "extraction_failures.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["session_id", "error"])
        writer.writeheader()
        writer.writerows(failures)
    (OUTPUT_DIR / "extraction_manifest.json").write_text(
        json.dumps({**runtime_manifest(), "temperature": LLM_TEMPERATURE}, indent=2),
        encoding="utf-8",
    )
    return evidence_path
