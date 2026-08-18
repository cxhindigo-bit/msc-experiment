import hashlib
import json
import re

from .config import DIALOGUE_TEMPERATURE, MAX_DIALOGUE_ATTEMPTS


# Require five tasks with five utterances each.
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
    answer = answer.strip()
    # Do not match a whole-number answer against a fraction component.
    if re.fullmatch(r"\d+", answer):
        return re.search(
            rf"(?<![\w/]){re.escape(answer)}(?![\w/])",
            text,
            flags=re.IGNORECASE,
        ) is not None
    return re.search(
        rf"(?<!\w){re.escape(answer)}(?!\w)", text, flags=re.IGNORECASE
    ) is not None


def _prompt(tasks, questions):
    generation_plan = []
    validation_plan = []
    for task in tasks:
        question = questions[task["question_id"]]
        correct = int(task["correct"]) == 1
        generation_plan.append({
            "task_id": task["task_id"],
            "question": question["text"],
            "initial_answer_status": "correct" if correct else "incorrect",
            "initial_answer_for_context": (
                question["correct_answer"] if correct else question["incorrect_answer"]
            ),
            "feedback_direction": question["feedback_hint"],
        })
        validation_plan.append({
            "question_text": question["text"],
            "initial_answer_status": "correct" if correct else "incorrect",
            "required_initial_answer": (
                question["correct_answer"] if correct
                else question["incorrect_answer"]
            ),
            "correct_answer": question["correct_answer"],
        })

    session_id = tasks[0]["session_id"]
    tutoring_styles = (
        "warm and encouraging",
        "calm and concise",
        "curious and Socratic",
        "supportive and reflective",
    )
    style = tutoring_styles[
        hashlib.sha256(session_id.encode("utf-8")).digest()[0] % len(tutoring_styles)
    ]
    system = (
        "Write one realistic one-to-one fraction tutoring session. "
        f"Use a {style} tutoring tone. Follow the supplied task plan exactly and return JSON only."
    )
    prompt = f"""Create one coherent tutoring session containing the five tasks below, in the given order.

Task plan:
{json.dumps(generation_plan, ensure_ascii=False, indent=2)}

For every task, write exactly five consecutive utterances in this order:
1. teacher asks the supplied question;
2. student gives the initial answer;
3. teacher gives one concise corrective prompt or check;
4. student gives the complete corrected answer after an incorrect initial answer, or a brief reason/check after a correct initial answer;
5. teacher confirms that complete answer and briefly states why it is correct.

Requirements:
- Preserve every number and the mathematical meaning of each supplied question.
- initial_answer_for_context is the fixed gold first answer. Use it only to make turns 3-5 respond naturally to that specific answer; the program replaces the second utterance itself.
- Treat initial_answer_status as fixed experimental information. Do not change whether the initial answer is correct.
- After an incorrect initial answer, follow feedback_direction in turn 3 without immediately stating the final answer; turn 4 must then give the complete correct answer to the original question.
- After a correct initial answer, use turn 3 for a brief check and turn 4 for a brief reason or check.
- In turn 5, repeat or explicitly identify the complete correct answer and name the correction principle (for example, common denominators, equal denominators, numerator order, or a common factor).
- Keep each utterance concise. Within this session, avoid repeating sentence openings such as "So", "Correct", "Exactly", or "Good job" across tasks; vary questions, acknowledgements, and confirmations naturally.
- Tasks 2-5 should continue the same session rather than restart the lesson.
- Do not mention task IDs, correctness labels, concepts, prompts, or experimental metadata in the dialogue text.

Return exactly this structure:
{{"tasks":[{{"task_id":"...","turns":[{{"speaker":"teacher","text":"..."}},{{"speaker":"student","text":"..."}},{{"speaker":"teacher","text":"..."}},{{"speaker":"student","text":"..."}},{{"speaker":"teacher","text":"..."}}]}}]}}"""
    return system, prompt, validation_plan


def _initial_answer_text(answer):
    """Build the fixed gold first answer without relying on model output."""
    return f"I think the answer is {answer}."


def _validate_generated(data, tasks, validation_plan, session_id):
    generated_tasks = data["tasks"]
    expected_ids = [task["task_id"] for task in tasks]
    if [task["task_id"] for task in generated_tasks] != expected_ids:
        raise ValueError(f"{session_id} changed the task order")

    speakers = ["teacher", "student", "teacher", "student", "teacher"]
    all_turns = []
    for task_index, (task, generated, planned) in enumerate(
        zip(tasks, generated_tasks, validation_plan)
    ):
        turns = generated["turns"]
        if len(turns) != 5 or [row["speaker"] for row in turns] != speakers:
            raise ValueError(f"{task['task_id']} must contain five alternating turns")
        if planned["question_text"].casefold() not in turns[0]["text"].casefold():
            raise ValueError(f"{task['task_id']} changed the supplied question")
        required = planned["required_initial_answer"]
        correct_answer = planned["correct_answer"]
        # A comparison question can already contain the correct fraction.
        answer_is_new_to_question = not _contains_answer(
            planned["question_text"], correct_answer
        )
        if (
            planned["initial_answer_status"] == "incorrect"
            and answer_is_new_to_question
            and _contains_answer(turns[2]["text"], correct_answer)
        ):
            raise ValueError(
                f"{task['task_id']} teacher revealed the correct answer before correction"
            )
        if (
            planned["initial_answer_status"] == "incorrect"
            and not _contains_answer(turns[3]["text"], correct_answer)
        ):
            original_text = turns[3]["text"].rstrip(". ")
            completion = f"So the complete answer is {correct_answer}."
            turns[3]["text"] = f"{original_text}. {completion}" if original_text else completion
        if "?" in turns[4]["text"]:
            turns[4]["text"] = f"That is correct: {correct_answer}."
        elif not _contains_answer(turns[4]["text"], correct_answer):
            turns[4]["text"] = (
                f"{turns[4]['text'].rstrip('. ')}. That is correct: {correct_answer}."
                if turns[4]["text"].strip()
                else f"That is correct: {correct_answer}."
            )
        turns[1]["text"] = _initial_answer_text(required)

        first_turn = task_index * 5 + 1
        all_turns.extend({
            "turn_id": f"{session_id}_T{first_turn + turn_index:03d}",
            "task_id": task["task_id"],
            "question_id": task["question_id"],
            "speaker": row["speaker"],
            "text": row["text"].strip(),
        } for turn_index, row in enumerate(turns))
    # Returned turns are written to data/raw_dialogues.jsonl by generate.write_dialogues().
    return all_turns


def generate_session_dialogue(tasks, questions):
    if len(tasks) != 5:
        raise ValueError("a session must contain exactly five tasks")
    system, prompt, validation_plan = _prompt(tasks, questions)
    session_id = tasks[0]["session_id"]
    last_error = None

    for attempt in range(MAX_DIALOGUE_ATTEMPTS):
        # Use a separate cache entry for each retry.
        cache_key = f"mastery-v5-{session_id}-attempt-{attempt}"
        try:
            data = ask_json(
                system,
                prompt,
                cache_key,
                response_schema=SESSION_SCHEMA,
                # Allow moderate wording variation.
                temperature=DIALOGUE_TEMPERATURE,
            )
            turns = _validate_generated(data, tasks, validation_plan, session_id)
            return turns, attempt
        except (KeyError, TypeError, ValueError) as error:
            last_error = error

    # Stop after the fixed retry limit.
    raise ValueError(
        f"{session_id} failed after {MAX_DIALOGUE_ATTEMPTS} attempts: {last_error}"
    )
