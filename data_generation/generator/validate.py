import csv
import json
from collections import Counter, defaultdict

from .config import CHECKPOINTS, CONCEPTS, N_LEARNERS, SESSION_DAYS, TASKS_PER_SESSION


def validate_gold(data):
    if len(data["profiles"]) != N_LEARNERS:
        raise ValueError("wrong learner count")
    if Counter(row["split"] for row in data["splits"]) != {
        "train": 144, "validation": 48, "test": 48
    }:
        raise ValueError("wrong split sizes")
    expected = N_LEARNERS * len(SESSION_DAYS) * TASKS_PER_SESSION
    if len(data["interactions"]) != expected:
        raise ValueError("wrong interaction count")
    for learner_id in {row["learner_id"] for row in data["profiles"]}:
        rows = [row for row in data["interactions"] if row["learner_id"] == learner_id]
        for checkpoint, minimum in zip(CHECKPOINTS, (2, 4)):
            counts = Counter(row["concept_id"] for row in rows if row["day"] <= checkpoint)
            if any(counts[concept] < minimum for concept in CONCEPTS):
                raise ValueError(f"insufficient concept coverage for {learner_id}")


def validate_dialogue_file(path):
    forbidden = {"correct", "concept_id", "mastery"}
    task_turns = defaultdict(list)
    turn_ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            session = json.loads(line)
            for turn in session["turns"]:
                if forbidden.intersection(turn):
                    raise ValueError("gold field found in raw dialogue")
                task_turns[turn["task_id"]].append(turn)
                turn_ids.append(turn["turn_id"])
    if len(turn_ids) != len(set(turn_ids)):
        raise ValueError("duplicate turn ID")
    expected_tasks = N_LEARNERS * len(SESSION_DAYS) * TASKS_PER_SESSION
    if len(task_turns) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} dialogue tasks, found {len(task_turns)}")
    speakers = ["teacher", "student", "teacher", "student", "teacher"]
    if any(len(turns) != 5 or [row["speaker"] for row in turns] != speakers for turns in task_turns.values()):
        raise ValueError("every task must contain five alternating turns")


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
