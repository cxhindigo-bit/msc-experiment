"""生成实验所需的结构化合成学习数据。

本文件为每名学习者设置固定的学习率和遗忘率，安排八次会话中的
四十项分数任务，并根据当前概念概率模拟首次回答、会话间遗忘和回答后的学习。
它还保存第 10 天和第 20 天的 F1-F8 参考概率，并在学习者层面划分训练集、
验证集和测试集。

Generate the structured synthetic learning data used by the experiment. The
module creates learner profiles, schedules forty fraction tasks across eight
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
    """将所有概念概率限制在实验规定的上下限内。 / Bound a probability to the experiment limits."""
    return min(MASTERY_MAX, max(MASTERY_MIN, value))


def _period_schedule(rng):
    """安排一个包含四次会话、二十项任务的学习阶段。 / Schedule one four-session, twenty-task period."""
    # 每 20 项任务包含八个概念各两项，以及四项运行时选择的薄弱概念任务
    # Each 20-task period contains two tasks per concept plus four weak-concept tasks selected at runtime.
    schedule = CONCEPTS * 2 + [None] * 4
    rng.shuffle(schedule)
    return schedule


def simulate_learner(learner_id, seed):
    """模拟一名学习者的四十项任务并返回资料、交互和两个日期的参考概率。"""
    rng = random.Random(seed)

    # 从规定范围内为该学习者抽取一次学习率，之后四十项任务都使用这个值。
    # Draw one learning rate for this learner and use it for all forty tasks.
    learning_rate = rng.uniform(*LEARNING_RATE_RANGE)

    # 从规定范围内为该学习者抽取一次会话间遗忘率，之后保持不变。
    # Draw one between-session forgetting rate for this learner and keep it fixed.
    forgetting_rate = rng.uniform(*FORGETTING_RATE_RANGE)

    # 分别初始化 F1-F8 的正确作答概率，每个概念拥有独立的初始值。
    # Initialize a separate correct-answer probability for each concept F1-F8.
    mastery = {concept: rng.uniform(*INITIAL_MASTERY) for concept in CONCEPTS}
    interactions, mastery_rows = [], []
    last_day = 0

    for period in range(2):
        # 每个阶段有二十项任务，即四次会话；两个阶段合计八次会话。
        # Each period has twenty tasks in four sessions; two periods make eight sessions.
        schedule = _period_schedule(rng)
        for period_task, planned_concept in enumerate(schedule):
            # divmod 将阶段内的任务位置转换为“第几次会话”和“会话内第几项任务”。
            # divmod converts the period position into a session offset and task offset.
            session_offset, task_offset = divmod(period_task, TASKS_PER_SESSION)
            session_number = period * 4 + session_offset + 1
            day = SESSION_DAYS[session_number - 1]
            if task_offset == 0:
                # 只在两次会话之间应用遗忘，不在同一会话的五项任务之间应用。
                # Apply forgetting between sessions, not between the five tasks in one session.
                elapsed = day - last_day
                decay = math.exp(-forgetting_rate * elapsed)
                mastery = {concept: _clip(value * decay) for concept, value in mastery.items()}
                last_day = day

            # None 表示薄弱概念任务：选择此时正确作答概率最低的概念。
            # None marks a weak-concept task: select the concept with the lowest current probability.
            concept = planned_concept or min(mastery, key=lambda key: (mastery[key], key))

            # 当前题库不区分难度；这里只从选定概念的三道题中随机抽取一道。
            # The current bank has no difficulty levels; select one of the concept's three questions.
            question = rng.choice(QUESTIONS_BY_CONCEPT[concept])
            mastery_before = mastery[concept]

            # 将当前概率作为伯努利试验的成功概率，生成首次回答的 0/1 标签。
            # Use the current probability as a Bernoulli success probability for the initial response.
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
            factor = 1.0 if correct else INCORRECT_LEARNING_FACTOR
            # 更新公式：m_after = m_before + learning_rate × factor × (1 - m_before)。
            # Update rule: m_after = m_before + learning_rate × factor × (1 - m_before).
            mastery[concept] = _clip(
                mastery_before + learning_rate * factor * (1.0 - mastery_before)
            )

        checkpoint = CHECKPOINTS[period]

        # 检查日期位于最近会话之后，因此保存前先计算这段间隔中的遗忘。
        # 这里只计算待保存的快照，不修改 mastery 或 last_day；下一阶段会从真实的
        # 最近会话日期继续计算，从而避免重复应用遗忘。
        # A checkpoint follows the latest session, so decay is applied to the saved snapshot.
        # mastery and last_day are left unchanged to prevent double decay in the next period.
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
    """把学习者随机划分为训练集、验证集和测试集，同一人的记录只属于一个集合。"""
    # 使用固定随机种子打乱全部 240 名学习者，再按 144:48:48（60:20:20）划分。
    # 这样既能复现同一划分，也能避免同一学习者的记录同时出现在训练集和测试集中。
    # Shuffle all 240 learners with the fixed seed, then split them 144:48:48 (60:20:20).
    # This makes the split reproducible and keeps each learner in exactly one partition.
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
    """生成全部学习者的结构化参考数据，但不生成自然语言对话。 / Generate all structured gold data, excluding dialogue."""
    # 全局 seed 控制数据划分；SEED + index 控制每名学习者的轨迹。
    # The global seed controls splitting; SEED + index controls each learner trajectory.
    rng = random.Random(SEED)
    learner_ids = [f"L{i:03d}" for i in range(1, N_LEARNERS + 1)]
    split_map = _split_map(learner_ids, rng)
    profiles, interactions, mastery_rows = [], [], []
    for index, learner_id in enumerate(learner_ids, 1):
        learner = simulate_learner(learner_id, SEED + index)
        profiles.append(learner["profile"])
        interactions.extend(learner["interactions"])
        mastery_rows.extend(learner["mastery"])
    # These collections are written to data/ by generate.write_gold().
    return {
        "profiles": profiles,  # data/learner_profiles.csv
        "splits": [
            {"learner_id": learner_id, "split": split_map[learner_id]}
            for learner_id in learner_ids
        ],  # data/splits.csv
        "interactions": interactions,  # data/gold_interactions.csv
        "mastery": mastery_rows,  # data/gold_mastery.csv
    }
