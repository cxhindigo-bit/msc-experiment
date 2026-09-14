"""Train official pyKT SimpleKT on train learners and early-stop on validation learners."""

import random

from ..settings import SEED, SIMPLEKT


def split_training_sequences(sequences):
    train = [row for row in sequences if row["split"] == "train"]
    validation = [row for row in sequences if row["split"] == "validation"]
    return train, validation


def snapshot_state_dict(state_dict):
    # Clone tensors so later epochs cannot overwrite the saved best parameters.
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _collate(rows):
    import torch

    max_steps = max(len(row["responses"]) - 1 for row in rows)

    def pad(values):
        return values + [0] * (max_steps - len(values))

    batch = {
        "qseqs": [], "cseqs": [], "rseqs": [],
        "shft_qseqs": [], "shft_cseqs": [], "shft_rseqs": [], "smasks": [],
    }
    for row in rows:
        steps = len(row["responses"]) - 1
        # Design item: SimpleKT training target
        # Current setting: Use the sequence up to the current interaction to predict the next response.
        batch["qseqs"].append(pad(row["question_ids"][:-1]))
        batch["cseqs"].append(pad(row["concept_ids"][:-1]))
        batch["rseqs"].append(pad(row["responses"][:-1]))
        batch["shft_qseqs"].append(pad(row["question_ids"][1:]))
        batch["shft_cseqs"].append(pad(row["concept_ids"][1:]))
        batch["shft_rseqs"].append(pad(row["responses"][1:]))
        # All formal sequences have equal length, so these masks contain only valid positions.
        batch["smasks"].append([True] * steps + [False] * (max_steps - steps))
    return {
        key: torch.tensor(value, dtype=torch.bool if key == "smasks" else torch.long)
        for key, value in batch.items()
    }


def _epoch(model, loader, device, optimizer=None):
    import torch
    import torch.nn.functional as functional

    model.train(optimizer is not None)
    losses = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for raw in loader:
            batch = {key: value.to(device) for key, value in raw.items()}
            if optimizer is not None:
                optimizer.zero_grad()
            predictions = model(batch)[:, 1:]
            mask = batch["smasks"]
            # Design item: Training loss
            # Current setting: Binary cross-entropy over valid next-response predictions.
            loss = functional.binary_cross_entropy(
                predictions[mask], batch["shft_rseqs"][mask].float()
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def train_model(sequences, question_count, concept_count):
    import numpy as np
    import torch
    # Design item: Model implementation
    # Current setting: Official pyKT simpleKT.
    # Source: https://github.com/pykt-team/pykt-toolkit/blob/f766468f1d3083f737e5fde22a17a08de0d52552/pykt/models/simplekt.py
    from pykt.models.simplekt import simpleKT
    from torch.utils.data import DataLoader

    train_rows, validation_rows = split_training_sequences(sequences)
    if not train_rows or not validation_rows:
        raise ValueError("both train and validation learners are required")

    # Design item: Training and epoch selection
    # Current setting: Fit on 144 training learners; select the saved epoch on 48 validation learners; do not use test labels.

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_rows,
        batch_size=SIMPLEKT["batch_size"],
        shuffle=True,
        collate_fn=_collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_rows,
        batch_size=SIMPLEKT["batch_size"],
        shuffle=False,
        collate_fn=_collate,
    )

    size = SIMPLEKT["embedding_size"]
    model = simpleKT(
        n_question=concept_count,
        n_pid=question_count,
        d_model=size,
        n_blocks=SIMPLEKT["n_blocks"],
        dropout=SIMPLEKT["dropout"],
        d_ff=size,
        seq_len=max(len(row["responses"]) for row in sequences) + 1,
        final_fc_dim=size,
        final_fc_dim2=size,
        num_attn_heads=SIMPLEKT["attention_heads"],
        emb_type="qid",
    ).to(device)
    # Design item: Optimizer
    # Current setting: Adam with learning rate 0.001.
    optimizer = torch.optim.Adam(model.parameters(), lr=SIMPLEKT["learning_rate"])

    # Design item: Validation and early stopping
    # Current setting: Save the lowest validation BCE loss and stop after five epochs without improvement.
    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(SIMPLEKT["max_epochs"]):
        _epoch(model, train_loader, device, optimizer)
        validation_loss = _epoch(model, validation_loader, device)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = snapshot_state_dict(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale == SIMPLEKT["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, device, best_loss
