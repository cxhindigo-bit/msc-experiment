"""数据生成命令入口：分别生成结构化参考数据和自然语言对话。

`gold` 调用模拟器并写出 CSV，不调用 LLM；`dialogues` 读取这些固定数据，
再调用 LM Studio 将每个五任务会话写成自然语言。该文件不运行预测管线。

Command entry point for data generation. `gold` writes structured CSV data
without an LLM; `dialogues` reads that fixed data and uses LM Studio to render
each five-task session. This module does not run either prediction pipeline.
"""

import argparse
import csv
import json
from collections import OrderedDict

from .generator.config import (
    DATA_DIR,
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
    """使用第一行的字段顺序写出一组同结构字典。 / Write uniform dictionaries using the first row's field order."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_gold():
    """生成、验证并保存不依赖 LLM 的结构化参考数据。 / Generate, validate, and save LLM-independent gold data."""
    output_dir = DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 所有题目顺序、首次回答标签和参考概率都在自然语言生成之前确定。
    # Task order, initial-response labels, and reference probabilities are fixed before dialogue generation.
    data = simulate_dataset()
    validate_gold(data)

    _write_csv(output_dir / "questions.csv", QUESTIONS)
    _write_csv(output_dir / "learner_profiles.csv", data["profiles"])
    _write_csv(output_dir / "splits.csv", data["splits"])
    _write_csv(output_dir / "gold_interactions.csv", data["interactions"])
    _write_csv(output_dir / "gold_mastery.csv", data["mastery"])

    # manifest 记录本次数据的规模和生成设置，便于检查和复现。
    # The manifest records scale and generation settings for audit and reproduction.
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
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return data


def _dialogue_job(args):
    """把一个已经确定的五任务会话交给对话生成器。 / Render one fixed five-task session."""
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
    """读取 gold 文件，为每个会话生成对话并保存 JSONL。 / Render every gold session and save JSONL."""
    from .generator.llm_client import llm_manifest

    output_dir = DATA_DIR

    # 该步骤只读取 write_gold() 的结果；它不会重新模拟学习者或改变 gold 标签。
    # This stage only reads write_gold() outputs; it does not resimulate learners or change gold labels.
    interactions = read_csv(output_dir / "gold_interactions.csv")
    questions = {row["id"]: row for row in QUESTIONS}
    grouped_tasks = OrderedDict()
    for task in interactions:
        # 输入 CSV 已按学习者、会话和任务排列，OrderedDict 保留这一生成顺序。
        # The CSV is already ordered; OrderedDict preserves learner, session, and task order.
        grouped_tasks.setdefault(task["session_id"], []).append(task)
    jobs = [
        (tasks, questions)
        for tasks in grouped_tasks.values()
    ]

    sessions = []
    retry_count = 0
    for index, job in enumerate(jobs, 1):
        # 每个会话通常请求一次；验证失败时最多尝试三次。
        # A session normally uses one request; validation failures allow up to three attempts.
        session = _dialogue_job(job)
        retry_count += session.pop("_retry_count")
        sessions.append(session)
        if index % 50 == 0 or index == len(jobs):
            print(f"sessions {index}/{len(jobs)}, retries {retry_count}")

    path = output_dir / "raw_dialogues.jsonl"

    # 全部会话完成后统一写出最终文件；若中途停止，重跑时已完成会话从缓存读取。
    # Write the final file after all sessions complete; cached sessions are reused after interruption.
    with path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session, ensure_ascii=False) + "\n")
    validate_dialogue_file(path)

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "dialogue_status": "complete",
            "dialogue_sessions": len(sessions),
            "dialogue_turns": sum(len(row["turns"]) for row in sessions),
            "dialogue_retries": retry_count,
            "dialogue_invalid_attempts": retry_count,
            **llm_manifest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main():
    """解析 plan、gold 或 dialogues 命令。 / Dispatch the plan, gold, or dialogues command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "gold", "dialogues"])
    args = parser.parse_args()

    tasks = N_LEARNERS * len(SESSION_DAYS) * TASKS_PER_SESSION
    if args.command == "plan":
        # plan 只显示预计规模，不创建或覆盖数据文件。
        # plan reports the expected scale without creating or overwriting data files.
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
