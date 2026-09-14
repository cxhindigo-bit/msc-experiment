"""Generate the structured synthetic learning data used by the experiment.

This module creates learner profiles, schedules forty fraction tasks across eight
sessions, simulates initial responses, between-session forgetting, and learning
updates, saves F1-F8 reference probabilities on Days 10 and 20, and creates
learner-level train, validation, and test splits. It does not generate dialogue.
"""

import math
import random

from .config import (
    CHECKPOINTS,
    CONCEPTS,
    FORGETTING_RATE_RANGE,
    INCORRECT_LEARNING_FACTOR,
    INITIAL_MASTERY,
    LEARNING_RATE_RANGE,
    MASTERY_MAX,
    MASTERY_MIN,
    N_LEARNERS,
    SEED,
    SESSION_DAYS,
    TASKS_PER_SESSION,
)
from .question_bank import QUESTIONS_BY_CONCEPT


def _clip(value):
    """Bound a probability to the experiment limits."""
    return min(MASTERY_MAX, max(MASTERY_MIN, value))


def _period_schedule(rng):
    """Schedule one four-session, twenty-task period."""
    # Design item: Task coverage in each 20-task period
    # Current setting: Two tasks for each of F1–F8 plus four weak-concept tasks; no prerequisite gating.
    schedule = CONCEPTS * 2 + [None] * 4
    # Design item: Task order
    # Current setting: Randomly order the 20 planned task slots.
    rng.shuffle(schedule)
    return schedule


def simulate_learner(learner_id, seed):
    """Simulate forty tasks and two checkpoint probabilities for one learner."""
    # Design item: Learner simulation seed
    # Current setting: Use the master seed plus the learner's numeric position.
    rng = random.Random(seed)

    # Draw each learner's stable update rates and starting concept probabilities once.
    learning_rate = rng.uniform(*LEARNING_RATE_RANGE)
    forgetting_rate = rng.uniform(*FORGETTING_RATE_RANGE)
    mastery = {concept: rng.uniform(*INITIAL_MASTERY) for concept in CONCEPTS}
    interactions, mastery_rows = [], []
    last_day = 0

    for period in range(2):
        # Build one 20-task period across four five-task sessions.
        schedule = _period_schedule(rng)
        for period_task, planned_concept in enumerate(schedule):
            session_offset, task_offset = divmod(period_task, TASKS_PER_SESSION)
            session_number = period * 4 + session_offset + 1
            day = SESSION_DAYS[session_number - 1]
            if task_offset == 0:
                elapsed = day - last_day
                # Design item: Between-session change
                # Current setting: m_previous × exp(-λ_i × elapsed days), clipped to 0.05–0.95.
                decay = math.exp(-forgetting_rate * elapsed)
                mastery = {concept: _clip(value * decay) for concept, value in mastery.items()}
                last_day = day

            # Design item: Weak-concept task
            # Current setting: Select the learner's lowest current concept probability when the slot is reached.
            concept = planned_concept or min(mastery, key=lambda key: (mastery[key], key))

            # Randomly select one of the three fixed questions for the chosen concept.
            question = rng.choice(QUESTIONS_BY_CONCEPT[concept])
            mastery_before = mastery[concept]

            # Design item: Initial-answer label
            # Current setting: One Bernoulli draw with probability m_before fixes correct as 0 or 1.
            correct = int(rng.random() < mastery_before)
            session_id = f"{learner_id}_S{session_number:02d}"
            task_number = task_offset + 1
            task_id = f"{session_id}_Q{task_number:02d}"
            source_turn = f"{session_id}_T{(task_number - 1) * 5 + 2:03d}"
            interactions.append({
                "learner_id": learner_id,
                "session_id": session_id,
                "task_id": task_id,
                "day": day,
                "question_id": question["id"],
                "concept_id": concept,
                "correct": correct,
                "source_turn": source_turn,
                "mastery_before": round(mastery_before, 8),
            })
            # Design item: Post-task probability update
            # Current setting: m_after = m_before + α_i × f × (1-m_before), with f=1.0 if correct and 0.20 otherwise.
            factor = 1.0 if correct else INCORRECT_LEARNING_FACTOR
            mastery[concept] = _clip(
                mastery_before + learning_rate * factor * (1.0 - mastery_before)
            )

        checkpoint = CHECKPOINTS[period]

        # Save F1–F8 after applying the same two-day gap at both checkpoints.
        decay = math.exp(-forgetting_rate * (checkpoint - last_day))
        for concept, value in mastery.items():
            mastery_rows.append({
                "learner_id": learner_id,
                "day": checkpoint,
                "concept_id": concept,
                "mastery": round(_clip(value * decay), 8),
            })

    return {
        "profile": {
            "learner_id": learner_id,
            "learning_rate": round(learning_rate, 8),
            "forgetting_rate": round(forgetting_rate, 8),
        },
        "interactions": interactions,
        "mastery": mastery_rows,
    }


def _split_map(learner_ids, rng):
    """Split learners into train, validation, and test sets without overlap."""
    # Design item: Learner-level data split
    # Current setting: 144 training, 48 validation, and 48 test learners (60:20:20).
    shuffled = list(learner_ids)
    rng.shuffle(shuffled)
    train_end = len(shuffled) * 3 // 5
    validation_end = len(shuffled) * 4 // 5
    split_map = {
        learner_id: "train" for learner_id in shuffled[:train_end]
    }
    split_map.update({
        learner_id: "validation"
        for learner_id in shuffled[train_end:validation_end]
    })
    split_map.update({
        learner_id: "test" for learner_id in shuffled[validation_end:]
    })
    return split_map


def simulate_dataset():
    """Generate all structured gold data, excluding dialogue."""
    # The master seed fixes the learner split; learner-specific seeds fix individual histories.
    rng = random.Random(SEED)
    learner_ids = [f"L{i:03d}" for i in range(1, N_LEARNERS + 1)]
    split_map = _split_map(learner_ids, rng)
    profiles, interactions, mastery_rows = [], [], []
    for index, learner_id in enumerate(learner_ids, 1):
        learner = simulate_learner(learner_id, SEED + index)
        profiles.append(learner["profile"])
        interactions.extend(learner["interactions"])
        mastery_rows.extend(learner["mastery"])
    return {
        "profiles": profiles,
        "splits": [
            {"learner_id": learner_id, "split": split_map[learner_id]}
            for learner_id in learner_ids
        ],
        "interactions": interactions,
        "mastery": mastery_rows,
    }
