"""Train SimpleKT and predict the eight fixed questions for test learners."""

import csv

from ..settings import CHECKPOINTS, CONCEPTS, OUTPUT_DIR, PROBE_QUESTIONS
from .sequences import prepare_sequences
from .train import train_model


def _single_batch(question_ids, concept_ids, responses, device):
    import torch

    return {
        "qseqs": torch.tensor([question_ids[:-1]], dtype=torch.long, device=device),
        "cseqs": torch.tensor([concept_ids[:-1]], dtype=torch.long, device=device),
        "rseqs": torch.tensor([responses[:-1]], dtype=torch.long, device=device),
        "shft_qseqs": torch.tensor([question_ids[1:]], dtype=torch.long, device=device),
        "shft_cseqs": torch.tensor([concept_ids[1:]], dtype=torch.long, device=device),
        "shft_rseqs": torch.tensor([responses[1:]], dtype=torch.long, device=device),
    }


def predict(model, sequences, question_map, concept_map, device):
    import torch

    rows = []
    with torch.no_grad():
        for sequence in sequences:
            if sequence["split"] != "test":
                continue
            for day in CHECKPOINTS:
                # Design item: Checkpoint history
                # Current setting: Day 10 uses 20 tasks and Day 20 uses all 40 tasks.
                keep = [index for index, value in enumerate(sequence["days"]) if value <= day]
                questions = [sequence["question_ids"][index] for index in keep]
                concepts = [sequence["concept_ids"][index] for index in keep]
                responses = [sequence["responses"][index] for index in keep]
                # Design item: Concept probability query
                # Current setting: Append the fixed test question for each F1–F8 concept in turn.
                for concept in CONCEPTS:
                    batch = _single_batch(
                        questions + [question_map[PROBE_QUESTIONS[concept]]],
                        concepts + [concept_map[concept]],
                        responses + [0],
                        device,
                    )
                    rows.append({
                        "learner_id": sequence["learner_id"],
                        "day": day,
                        "concept_id": concept,
                        "probability": round(float(model(batch)[0, -1].cpu()), 8),
                    })
    return rows


def run(label_source="llm"):
    sequences, question_map, concept_map = prepare_sequences(label_source)
    model, device, validation_loss = train_model(
        sequences, len(question_map), len(concept_map)
    )
    rows = predict(model, sequences, question_map, concept_map, device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        "llm_simplekt_predictions.csv"
        if label_source == "llm"
        else "gold_simplekt_predictions.csv"
    )
    path = OUTPUT_DIR / filename
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["learner_id", "day", "concept_id", "probability"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{label_source}-SimpleKT] best validation loss={validation_loss:.6f}")
    return path
