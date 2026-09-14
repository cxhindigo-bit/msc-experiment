"""Generate structured gold data and OpenAI-compatible dialogue data."""

import argparse
import csv
import json
from collections import OrderedDict

from .generator.config import (
    DATA_DIR,
    DIALOGUE_INITIAL_ANSWER_MODE,
    DIALOGUE_THINKING,
    FORGETTING_RATE_RANGE,
    LEARNING_RATE_RANGE,
    N_LEARNERS,
    SEED,
    SESSION_DAYS,
    TASKS_PER_SESSION,
)
from .generator.question_bank import QUESTIONS
from .generator.simulator import simulate_dataset
from .generator.validate import read_csv, validate_dialogue_file, validate_gold


def _write_csv(path, rows):
    """Write rows using the first row's field order."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_gold():
    """Generate and save LLM-independent gold data."""
    output_dir = DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Design item: Gold-first data construction
    # Current setting: Fix questions, initial-answer labels, and reference probabilities before generating dialogue.
    data = simulate_dataset()
    validate_gold(data)

    _write_csv(output_dir / "questions.csv", QUESTIONS)
    _write_csv(output_dir / "learner_profiles.csv", data["profiles"])
    _write_csv(output_dir / "splits.csv", data["splits"])
    _write_csv(output_dir / "gold_interactions.csv", data["interactions"])
    _write_csv(output_dir / "gold_mastery.csv", data["mastery"])

    # Design item: Generation record
    # Current setting: Save the seed, scale, parameter ranges, and dialogue status in the manifest.
    manifest = {
        "seed": SEED,
        "learners": len(data["profiles"]),
        "sessions": N_LEARNERS * len(SESSION_DAYS),
        "interactions": len(data["interactions"]),
        "checkpoints": [10, 20],
        "turns_per_task": 5,
        "llm_calls_required": N_LEARNERS * len(SESSION_DAYS),
        "learning_rate_range": LEARNING_RATE_RANGE,
        "forgetting_rate_range": FORGETTING_RATE_RANGE,
        "dialogue_status": "not_generated",
        "llm_thinking": DIALOGUE_THINKING,
        "dialogue_initial_answer_mode": DIALOGUE_INITIAL_ANSWER_MODE,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return data


def _dialogue_job(args):
    """Render one fixed five-task session."""
    from .generator.dialogue import generate_session_dialogue

    tasks, questions = args
    first = tasks[0]
    turns, retry_count = generate_session_dialogue(tasks, questions)
    return {
        "learner_id": first["learner_id"],
        "session_id": first["session_id"],
        "day": int(first["day"]),
        "turns": turns,
        "_retry_count": retry_count,
    }


def write_dialogues():
    """Render every gold session and save JSONL."""
    from .generator.llm_client import llm_manifest

    output_dir = DATA_DIR

    # Dialogue generation reads the completed gold interactions without regenerating them.
    interactions = read_csv(output_dir / "gold_interactions.csv")
    questions = {row["id"]: row for row in QUESTIONS}
    grouped_tasks = OrderedDict()
    for task in interactions:
        # Preserve the session and task order already stored by the simulator.
        grouped_tasks.setdefault(task["session_id"], []).append(task)
    jobs = [
        (tasks, questions)
        for tasks in grouped_tasks.values()
    ]

    sessions = []
    retry_count = 0
    for index, job in enumerate(jobs, 1):
        session = _dialogue_job(job)
        retry_count += session.pop("_retry_count")
        sessions.append(session)
        if index % 50 == 0 or index == len(jobs):
            print(f"sessions {index}/{len(jobs)}, retries {retry_count}")

    path = output_dir / "raw_dialogues.jsonl"

    # Design item: Dialogue file structure
    # Current setting: 1,920 sessions, five tasks per session, and five utterances per task.
    with path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session, ensure_ascii=False) + "\n")
    # Design item: Final dialogue acceptance
    # Current setting: Validate the complete dialogue file against the gold tasks before marking it complete.
    validate_dialogue_file(path, interactions=interactions, questions=questions)

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "dialogue_status": "complete",
            "dialogue_sessions": len(sessions),
            "dialogue_turns": sum(len(row["turns"]) for row in sessions),
            "dialogue_retries": retry_count,
            "dialogue_invalid_attempts": retry_count,
            "dialogue_initial_answer_mode": DIALOGUE_INITIAL_ANSWER_MODE,
            **llm_manifest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main():
    """Dispatch the plan, gold, or dialogues command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "gold", "dialogues"])
    args = parser.parse_args()

    tasks = N_LEARNERS * len(SESSION_DAYS) * TASKS_PER_SESSION
    if args.command == "plan":
        print(
            {
                "learners": N_LEARNERS,
                "sessions": N_LEARNERS * len(SESSION_DAYS),
                "tasks": tasks,
                "llm_calls": N_LEARNERS * len(SESSION_DAYS),
            }
        )
    elif args.command == "gold":
        write_gold()
        print(f"gold data written to {DATA_DIR}")
    elif args.command == "dialogues":
        print(write_dialogues())


if __name__ == "__main__":
    main()
