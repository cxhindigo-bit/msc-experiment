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


def _contains_answer(text, answer):
    # Match text answers case-insensitively and numbers strictly.
    import re

    answer = answer.strip()
    if re.fullmatch(r"\d+", answer):
        return re.search(rf"(?<![\w/]){re.escape(answer)}(?![\w/])", text) is not None
    return re.search(
        rf"(?<!\w){re.escape(answer)}(?!\w)", text, flags=re.IGNORECASE
    ) is not None


def validate_dialogue_file(path, interactions=None, questions=None):
    # Optionally validate each final dialogue against gold constraints.
    forbidden = {"correct", "concept_id", "mastery"}
    task_turns = defaultdict(list)
    turn_ids = []
    # Read every session and turn from the JSONL file.
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            session = json.loads(line)
            for turn in session["turns"]:
                # Raw dialogue must not expose internal gold fields.
                if forbidden.intersection(turn):
                    raise ValueError("gold field found in raw dialogue")
                task_turns[turn["task_id"]].append(turn)
                turn_ids.append(turn["turn_id"])
    # Each turn ID must be unique across the complete dataset.
    if len(turn_ids) != len(set(turn_ids)):
        raise ValueError("duplicate turn ID")
    expected_tasks = N_LEARNERS * len(SESSION_DAYS) * TASKS_PER_SESSION
    # Require the planned number of dialogue tasks.
    if len(task_turns) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} dialogue tasks, found {len(task_turns)}")
    speakers = ["teacher", "student", "teacher", "student", "teacher"]
    # Every task has five turns in teacher-student-teacher-student-teacher order.
    if any(len(turns) != 5 or [row["speaker"] for row in turns] != speakers for turns in task_turns.values()):
        raise ValueError("every task must contain five alternating turns")
    if interactions is None or questions is None:
        return
    expected = {row["task_id"]: row for row in interactions}
    for task_id, turns in task_turns.items():
        task = expected.get(task_id)
        if task is None:
            raise ValueError(f"unexpected dialogue task {task_id}")
        question = questions[task["question_id"]]
        # The teacher must preserve the supplied question.
        if question["text"].casefold() not in turns[0]["text"].casefold():
            raise ValueError(f"{task_id} changed the supplied question")
        correct_answer = question["correct_answer"]
        initial_answer = correct_answer if int(task["correct"]) else question["incorrect_answer"]
        # The student initial answer must retain the planned gold answer.
        if not _contains_answer(turns[1]["text"], initial_answer):
            raise ValueError(f"{task_id} lost its gold initial answer")
        if int(task["correct"]) == 0:
            answer_is_new_to_question = not _contains_answer(
                question["text"], correct_answer
            )
            # Feedback must not reveal a new correct answer before student correction.
            if answer_is_new_to_question and _contains_answer(
                turns[2]["text"], correct_answer
            ):
                raise ValueError(f"{task_id} reveals the correct answer before correction")
            # The correction turn must state the complete correct answer.
            if not _contains_answer(turns[3]["text"], correct_answer):
                raise ValueError(f"{task_id} lacks the corrected final answer")
        # The final teacher turn must confirm the answer rather than ask again.
        if "?" in turns[4]["text"] or not _contains_answer(turns[4]["text"], correct_answer):
            raise ValueError(f"{task_id} lacks a closing teacher confirmation")


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
