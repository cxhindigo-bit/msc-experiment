"""Coverage-first, learner-level evaluation."""

import csv
import json
import math
import random

from ..settings import BOOTSTRAP_SAMPLES, CHECKPOINTS, DATA_DIR, OUTPUT_DIR, SEED


def _value(row):
    return float(row["probability"] if "probability" in row else row["mastery"])


def prediction_map(rows):
    # Convert each learner–day–concept row to one validated probability.
    result = {}
    for row in rows:
        key = row["learner_id"], int(row["day"]), row["concept_id"]
        if key in result:
            raise ValueError(f"duplicate prediction: {key}")
        value = _value(row)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid probability for {key}")
        result[key] = value
    return result


def prediction_coverage(rows, gold):
    # Design item: Prediction coverage
    # Current setting: Require the exact 768 learner–day–concept keys before calculating MAE or RMSE.
    predictions = prediction_map(rows)
    expected, actual = set(gold), set(predictions)
    learners = {key[0] for key in expected}
    complete = {
        learner for learner in learners
        if all(key in actual for key in expected if key[0] == learner)
    }
    return {
        "valid_predictions": len(expected & actual),
        "expected_predictions": len(expected),
        "missing_predictions": len(expected - actual),
        "extra_predictions": len(actual - expected),
        "complete_learners": len(complete),
        "expected_learners": len(learners),
        "coverage": len(expected & actual) / len(expected),
    }


def _mean_ci(values, seed):
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    # Design item: Bootstrap interval calculation
    # Current setting: Resample learner-level values 2,000 times and use the 2.5th and 97.5th percentiles of the means.
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def state_metrics(rows, gold, learners):
    # Design item: Probability-error metrics
    # Current setting: Compare all methods with the same reference values; calculate learner-level MAE and RMSE before averaging learners.
    predictions = prediction_map(rows)
    expected = {key: value for key, value in gold.items() if key[0] in learners}
    if set(predictions) != set(expected):
        raise ValueError("complete predictions are required before error metrics")

    summaries, learner_rows = [], []
    days = sorted({key[1] for key in expected})
    for scope_index, day in enumerate(days + ["all"]):
        scoped = []
        for learner in sorted(learners):
            # Use eight concept values for one day or 16 values for the combined result.
            keys = [
                key for key in expected
                if key[0] == learner and (day == "all" or key[1] == day)
            ]
            errors = [predictions[key] - expected[key] for key in keys]
            row = {
                "learner_id": learner,
                "day": day,
                "predictions": len(errors),
                "mae": sum(abs(error) for error in errors) / len(errors),
                "rmse": math.sqrt(sum(error ** 2 for error in errors) / len(errors)),
            }
            scoped.append(row)
            learner_rows.append(row)
        summary = {
            "day": day,
            "learners": len(scoped),
            "predictions": sum(row["predictions"] for row in scoped),
        }
        for metric_index, metric in enumerate(("mae", "rmse")):
            values = [row[metric] for row in scoped]
            low, high = _mean_ci(values, SEED + scope_index * 10 + metric_index)
            summary.update({
                metric: sum(values) / len(values),
                f"{metric}_ci_low": low,
                f"{metric}_ci_high": high,
            })
        summaries.append(summary)
    return summaries, learner_rows


def classification_metrics(extracted_rows, gold_rows):
    # Design item: Initial-answer classification evaluation
    # Current setting: Report task/session coverage, accuracy, and class-specific precision, recall, and F1 against 9,600 gold labels.
    def key(row):
        return row["learner_id"], row["session_id"], row["task_id"]

    gold = {key(row): int(row["correct"]) for row in gold_rows}
    extracted = {key(row): int(row["correct"]) for row in extracted_rows}
    matched = sorted(set(gold) & set(extracted))
    expected_sessions = {(row["learner_id"], row["session_id"]) for row in gold_rows}
    complete_sessions = {
        session for session in expected_sessions
        if all(item in extracted for item in gold if item[:2] == session)
    }
    result = {
        "matched_tasks": len(matched),
        "expected_tasks": len(gold),
        "task_coverage": len(matched) / len(gold),
        "complete_sessions": len(complete_sessions),
        "expected_sessions": len(expected_sessions),
        "session_coverage": len(complete_sessions) / len(expected_sessions),
        "accuracy": (
            sum(extracted[item] == gold[item] for item in matched) / len(matched)
            if matched else 0.0
        ),
    }
    for label, name in ((1, "correct"), (0, "incorrect")):
        true_positive = sum(
            gold[item] == label and extracted[item] == label for item in matched
        )
        false_positive = sum(
            gold[item] != label and extracted[item] == label for item in matched
        )
        false_negative = sum(
            gold[item] == label and extracted[item] != label for item in matched
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        result[f"{name}_precision"] = precision
        result[f"{name}_recall"] = recall
        result[f"{name}_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return result


def paired_metrics(learner_rows, left_system, right_system):
    # Design item: Paired method comparison
    # Current setting: Subtract learner-level MAE and RMSE for methods evaluated on the same 48 learners.
    lookup = {
        (row["system"], row["learner_id"], row["day"]): row
        for row in learner_rows
    }
    days = list(dict.fromkeys(
        row["day"] for row in learner_rows if row["system"] == left_system
    ))
    summaries = []
    for day_index, day in enumerate(days):
        learners = sorted(
            row["learner_id"] for row in learner_rows
            if row["system"] == left_system
            and row["day"] == day
            and (right_system, row["learner_id"], day) in lookup
        )
        if not learners:
            continue
        summary = {
            "left_system": left_system,
            "right_system": right_system,
            "day": day,
            "learners": len(learners),
        }
        for metric_index, metric in enumerate(("mae", "rmse")):
            differences = [
                lookup[(left_system, learner, day)][metric]
                - lookup[(right_system, learner, day)][metric]
                for learner in learners
            ]
            low, high = _mean_ci(
                differences, SEED + 500 + day_index * 10 + metric_index
            )
            summary[f"{metric}_difference"] = sum(differences) / len(differences)
            summary[f"{metric}_difference_ci_low"] = low
            summary[f"{metric}_difference_ci_high"] = high
        summaries.append(summary)
    return summaries


def _read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    # Design item: Common evaluation cohort
    # Current setting: Use the same 48 test learners for every probability-output condition.
    splits = {row["learner_id"]: row["split"] for row in _read_csv(DATA_DIR / "splits.csv")}
    test_ids = {learner for learner, split in splits.items() if split == "test"}
    # Design item: Reference probability
    # Current setting: Use the simulator's learner–day–concept value; questions within one concept have no separate difficulty.
    gold_rows = _read_csv(DATA_DIR / "gold_mastery.csv")
    gold = {
        (row["learner_id"], int(row["day"]), row["concept_id"]): float(row["mastery"])
        for row in gold_rows if row["learner_id"] in test_ids
    }
    # Design item: Compared probability outputs
    # Current setting: Pure-LLM and LLM-SimpleKT are main pipelines; gold-SimpleKT is a diagnostic condition.
    systems = {
        "Pure-LLM": OUTPUT_DIR / "pure_llm_predictions.csv",
        "LLM-SimpleKT": OUTPUT_DIR / "llm_simplekt_predictions.csv",
        "gold-SimpleKT": OUTPUT_DIR / "gold_simplekt_predictions.csv",
    }
    coverage_rows, metric_rows, learner_rows = [], [], []
    for name, path in systems.items():
        rows = _read_csv(path) if path.exists() else []
        coverage = prediction_coverage(rows, gold)
        coverage_rows.append({"system": name, **coverage})
        # Do not calculate error for an incomplete condition.
        if coverage["coverage"] != 1 or coverage["extra_predictions"]:
            continue
        summaries, learners = state_metrics(rows, gold, test_ids)
        metric_rows.extend({"system": name, **row} for row in summaries)
        learner_rows.extend({"system": name, **row} for row in learners)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "prediction_coverage.csv", coverage_rows)
    _write_csv(OUTPUT_DIR / "state_metrics.csv", metric_rows)
    _write_csv(OUTPUT_DIR / "learner_state_metrics.csv", learner_rows)

    paired_rows = paired_metrics(learner_rows, "Pure-LLM", "LLM-SimpleKT")
    # Design item: Effect of LLM classification errors
    # Current setting: Compare LLM-SimpleKT minus gold-SimpleKT on matched learners.
    paired_rows.extend(
        paired_metrics(learner_rows, "LLM-SimpleKT", "gold-SimpleKT")
    )
    _write_csv(OUTPUT_DIR / "paired_state_metrics.csv", paired_rows)

    extraction_path = OUTPUT_DIR / "extracted_evidence.jsonl"
    if extraction_path.exists():
        extracted = [
            json.loads(line)
            for line in extraction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        classification = classification_metrics(
            extracted, _read_csv(DATA_DIR / "gold_interactions.csv")
        )
        _write_csv(OUTPUT_DIR / "classification_metrics.csv", [classification])
    return metric_rows
