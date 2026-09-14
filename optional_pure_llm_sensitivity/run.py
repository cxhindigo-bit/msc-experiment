"""Run the two optional Pure-LLM sensitivity conditions without changing the main experiment."""

import argparse
import csv
import json
import math
from pathlib import Path

from experiment.settings import (
    CHECKPOINTS,
    CONCEPT_NAMES,
    CONCEPTS,
    DATA_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    MAX_LLM_ATTEMPTS,
    OUTPUT_DIR,
    PROBE_QUESTIONS,
)


# Design item: Optional Pure-LLM sensitivity conditions
# Current setting: Use the improved prompt with gpt-4.1-mini and gpt-4.1 at temperature 0.0.
CONDITIONS = {
    "improved-mini": "gpt-4.1-mini",
    "improved-full": "gpt-4.1",
}
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
TEMPERATURE = 0.0
PROMPT_VERSION = "improved-evidence-prompt-v1"
EXPERIMENT_DIR = OUTPUT_DIR / "optional_pure_llm_sensitivity"
SYSTEM = "Estimate first-attempt correctness probabilities from tutoring evidence. Return JSON only."


def _read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_improved_prompt(learner_id, day, sessions, question_text):
    fixed_questions = "\n".join(
        f"- {concept}: {CONCEPT_NAMES[concept]}; {PROBE_QUESTIONS[concept]}: "
        f"{question_text[PROBE_QUESTIONS[concept]]}"
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

For each fixed question, estimate the probability that the learner would answer
correctly on the first attempt without teacher help.

Use the evidence in this way:
1. Treat the first student answer to each task as the main correctness evidence.
2. A correct answer produced after a teacher hint is not independent evidence of
   an unaided correct answer. It may show partial learning, but do not treat it as
   equivalent to a correct first answer.
3. Judge each concept from its own relevant history. Do not give all concepts a
   similar score from the overall tone of the dialogue.
4. Consider repeated evidence and its order. Several recent correct first answers
   support a higher probability than one corrected answer after a hint.
5. Use only dialogue dated on or before the prediction day.

FIXED QUESTIONS
{fixed_questions}

VISIBLE DIALOGUE
{history}
"""


def _validate(values):
    if not isinstance(values, dict) or set(values) != set(CONCEPTS):
        raise ValueError("the response must contain exactly F1-F8")
    result = {}
    for concept in CONCEPTS:
        value = values[concept]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{concept} must be numeric")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{concept} must be within [0, 1]")
        result[concept] = float(value)
    return result


def _request(condition, model, cache_key, prompt):
    from openai import OpenAI

    cache_path = EXPERIMENT_DIR / "cache" / condition / PROMPT_VERSION / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": f"pure_llm_{condition.replace('-', '_')}",
                "strict": True,
                "schema": SCHEMA,
            },
        },
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("the model returned empty content")
    data = json.loads(content)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def run_condition(condition, limit_learners=0):
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    model = CONDITIONS[condition]
    sessions = [
        json.loads(line)
        for line in (DATA_DIR / "raw_dialogues.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_ids = [
        row["learner_id"] for row in _read_csv(DATA_DIR / "splits.csv")
        if row["split"] == "test"
    ]
    if limit_learners:
        test_ids = test_ids[:limit_learners]
    question_text = {row["id"]: row["text"] for row in _read_csv(DATA_DIR / "questions.csv")}

    predictions, failures = [], []
    for learner_id in test_ids:
        for day in CHECKPOINTS:
            visible = [
                session for session in sessions
                if session["learner_id"] == learner_id and int(session["day"]) <= day
            ]
            prompt = build_improved_prompt(learner_id, day, visible, question_text)
            error = None
            for attempt in range(MAX_LLM_ATTEMPTS):
                try:
                    data = _request(
                        condition,
                        model,
                        f"{learner_id}-day-{day}-attempt-{attempt}",
                        prompt,
                    )
                    values = _validate(data["probabilities"])
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

    condition_dir = EXPERIMENT_DIR / condition
    _write_csv(
        condition_dir / "predictions.csv",
        ["learner_id", "day", "concept_id", "probability"],
        predictions,
    )
    _write_csv(
        condition_dir / "failures.csv",
        ["learner_id", "day", "error"],
        failures,
    )
    (condition_dir / "manifest.json").write_text(
        json.dumps(
            {
                "condition": condition,
                "model": model,
                "base_url": LLM_BASE_URL,
                "temperature": TEMPERATURE,
                "prompt": PROMPT_VERSION,
                "test_learners": len(test_ids),
                "checkpoints": CHECKPOINTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return condition_dir / "predictions.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=list(CONDITIONS))
    parser.add_argument("--all", action="store_true", help="run both new conditions")
    parser.add_argument("--smoke", type=int, default=0, help="limit the number of test learners")
    args = parser.parse_args()
    if not args.all and not args.condition:
        parser.error("choose --condition or --all")
    selected = list(CONDITIONS) if args.all else [args.condition]
    for condition in selected:
        path = run_condition(condition, args.smoke)
        print(f"[{condition}] wrote {path}")


if __name__ == "__main__":
    main()
