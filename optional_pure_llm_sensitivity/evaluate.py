"""Compare the optional Pure-LLM conditions with the completed main experiment."""

import csv

from experiment.evaluation.evaluate import paired_metrics, prediction_coverage, state_metrics
from experiment.settings import DATA_DIR, OUTPUT_DIR

from .run import CONDITIONS, EXPERIMENT_DIR


def _read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    splits = {row["learner_id"]: row["split"] for row in _read_csv(DATA_DIR / "splits.csv")}
    test_ids = {learner for learner, split in splits.items() if split == "test"}
    gold = {
        (row["learner_id"], int(row["day"]), row["concept_id"]): float(row["mastery"])
        for row in _read_csv(DATA_DIR / "gold_mastery.csv")
        if row["learner_id"] in test_ids
    }
    systems = {
        "original-mini": OUTPUT_DIR / "pure_llm_predictions.csv",
        **{
            condition: EXPERIMENT_DIR / condition / "predictions.csv"
            for condition in CONDITIONS
        },
    }

    coverage_rows, metric_rows, learner_rows = [], [], []
    for name, path in systems.items():
        rows = _read_csv(path) if path.exists() else []
        coverage = prediction_coverage(rows, gold)
        coverage_rows.append({"condition": name, **coverage})
        if coverage["coverage"] != 1 or coverage["extra_predictions"]:
            continue
        summaries, learners = state_metrics(rows, gold, test_ids)
        metric_rows.extend({"condition": name, **row} for row in summaries)
        learner_rows.extend({"system": name, **row} for row in learners)

    paired_rows = []
    for condition in CONDITIONS:
        paired_rows.extend(paired_metrics(learner_rows, condition, "original-mini"))
    paired_rows.extend(paired_metrics(learner_rows, "improved-full", "improved-mini"))

    _write_csv(EXPERIMENT_DIR / "coverage.csv", coverage_rows)
    _write_csv(EXPERIMENT_DIR / "metrics.csv", metric_rows)
    _write_csv(EXPERIMENT_DIR / "learner_metrics.csv", learner_rows)
    _write_csv(EXPERIMENT_DIR / "paired_comparisons.csv", paired_rows)


if __name__ == "__main__":
    run()
