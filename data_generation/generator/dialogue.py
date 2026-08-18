import json
import re

from .config import DIALOGUE_TEMPERATURE, MAX_DIALOGUE_ATTEMPTS

# LM Studio 使用该 Schema 强制一次返回五项任务，每项正好五次发言。
# LM Studio uses this schema to return five tasks with exactly five utterances each.
SESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "turns": {
                        "type": "array",
                        "minItems": 5,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "properties": {
                                "speaker": {
                                    "type": "string",
                                    "enum": ["teacher", "student"],
                                },
                                "text": {"type": "string"},
                            },
                            "required": ["speaker", "text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["task_id", "turns"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}


def ask_json(system, prompt, cache_key, response_schema, temperature):
    from .llm_client import ask_json as call

    return call(
        system,
        prompt,
        cache_key,
        response_schema=response_schema,
        temperature=temperature,
    )


def _contains_answer(text, answer):
    return re.search(
        rf"(?<!\w){re.escape(answer)}(?!\w)", text, flags=re.IGNORECASE
    ) is not None


def _prompt(tasks, questions):
    task_plan = []
    for task in tasks:
        question = questions[task["question_id"]]
        correct = int(task["correct"]) == 1
        task_plan.append({
            "task_id": task["task_id"],
            "question": question["text"],
            "initial_answer_status": "correct" if correct else "incorrect",
            "required_initial_answer": (
                question["correct_answer"] if correct
                else question["incorrect_answer"]
            ),
            "feedback_direction": question["feedback_hint"],
        })

    system = (
        "Write one realistic one-to-one fraction tutoring session. "
        "Follow the supplied task plan exactly and return JSON only."
    )
    prompt = f"""Create one coherent tutoring session containing the five tasks below, in the given order.

Task plan:
{json.dumps(task_plan, ensure_ascii=False, indent=2)}

For every task, write exactly five consecutive utterances in this order:
1. teacher asks the supplied question;
2. student gives the initial answer;
3. teacher responds;
4. student replies or revises the answer;
5. teacher closes the task briefly.

Requirements:
- Preserve every number and the mathematical meaning of each supplied question.
- The first student utterance must contain required_initial_answer exactly, but the surrounding wording should sound natural.
- Treat initial_answer_status as fixed experimental information. Do not change whether the initial answer is correct.
- After an incorrect initial answer, follow feedback_direction without immediately stating the final answer.
- After a correct initial answer, confirm it and ask for a brief reason or check.
- Keep each utterance concise. Vary wording naturally and avoid repeating greetings or stock phrases across the five tasks.
- Tasks 2-5 should continue the same session rather than restart the lesson.
- Do not mention task IDs, correctness labels, concepts, prompts, or experimental metadata in the dialogue text.

Return exactly this structure:
{{"tasks":[{{"task_id":"...","turns":[{{"speaker":"teacher","text":"..."}},{{"speaker":"student","text":"..."}},{{"speaker":"teacher","text":"..."}},{{"speaker":"student","text":"..."}},{{"speaker":"teacher","text":"..."}}]}}]}}"""
    return system, prompt, task_plan


def _validate_generated(data, tasks, task_plan, session_id):
    generated_tasks = data["tasks"]
    expected_ids = [task["task_id"] for task in tasks]
    if [task["task_id"] for task in generated_tasks] != expected_ids:
        raise ValueError(f"{session_id} changed the task order")

    speakers = ["teacher", "student", "teacher", "student", "teacher"]
    all_turns = []
    for task_index, (task, generated, planned) in enumerate(
        zip(tasks, generated_tasks, task_plan)
    ):
        turns = generated["turns"]
        if len(turns) != 5 or [row["speaker"] for row in turns] != speakers:
            raise ValueError(f"{task['task_id']} must contain five alternating turns")
        required = planned["required_initial_answer"]
        if not _contains_answer(turns[1]["text"], required):
            raise ValueError(f"{task['task_id']} changed the required initial answer")

        first_turn = task_index * 5 + 1
        all_turns.extend({
            "turn_id": f"{session_id}_T{first_turn + turn_index:03d}",
            "task_id": task["task_id"],
            "question_id": task["question_id"],
            "speaker": row["speaker"],
            "text": row["text"].strip(),
        } for turn_index, row in enumerate(turns))
    return all_turns


def generate_session_dialogue(tasks, questions):
    if len(tasks) != 5:
        raise ValueError("a session must contain exactly five tasks")
    system, prompt, task_plan = _prompt(tasks, questions)
    session_id = tasks[0]["session_id"]
    last_error = None

    for attempt in range(MAX_DIALOGUE_ATTEMPTS):
        # 每次尝试使用独立缓存键。不合格的 JSON 仍保留供检查，
        # 下一次尝试不会再读取同一个无效缓存。
        # Each attempt has a separate cache key. Invalid JSON remains available
        # for inspection, while the next attempt cannot reuse the same invalid entry.
        cache_key = f"mastery-v5-{session_id}-attempt-{attempt}"
        try:
            data = ask_json(
                system,
                prompt,
                cache_key,
                response_schema=SESSION_SCHEMA,
                # 0.7 采用 LM Studio 官方结构化输出示例中的设置。
                # It permits moderate wording variation and is not claimed as optimal.
                # https://github.com/lmstudio-ai/docs/blob/b02d17517b73c51f520cd5129855cdf30e0728f7/1_developer/3_openai-compat/structured-output.md
                temperature=DIALOGUE_TEMPERATURE,
            )
            turns = _validate_generated(data, tasks, task_plan, session_id)
            return turns, attempt
        except (KeyError, TypeError, ValueError) as error:
            last_error = error

    # 最多三次是硬性上限；到达上限后停止并报告会话，不无限循环。
    # The fixed attempt limit prevents an infinite retry loop.
    raise ValueError(
        f"{session_id} failed after {MAX_DIALOGUE_ATTEMPTS} attempts: {last_error}"
    )
