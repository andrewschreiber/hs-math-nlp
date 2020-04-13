import os
from pathlib import Path
from google.cloud import storage
import torch

BUCKET_NAME = "math-checkpoints-data"


def save_checkpoint_to_bucket(state, preempted, exp, path):
    if not os.path.isdir(path):
        os.makedirs(path)

    filename = f"{exp}_latest_checkpoint.pth"

    filepath = Path(path) / filename
    torch.save(state, filepath)

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)

    blob.upload_from_filename(filepath)

    print(f"File {filename} uploaded to {filepath}.")


def load_latest_checkpoint_from_bucket(exp, model, optimizer):
    source_blob_name = f"{exp}_latest_checkpoint.pth"
    destination_file_name = Path(".") / source_blob_name
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
    except Exception:
        print(f"No file {source_blob_name} found in bucket.")
        return None
    return restore_checkpoint(destination_file_name, model, optimizer)


def rotating_save_checkpoint(state, prefix, path="./checkpoints", nb=5):
    if not os.path.isdir(path):
        os.makedirs(path)
    filenames = []
    first_empty = None
    best_filename = Path(path) / f"{prefix}_best.pth"
    torch.save(state, best_filename)
    for i in range(nb):
        filename = Path(path) / f"{prefix}_{i}.pth"
        if not os.path.isfile(filename) and first_empty is None:
            first_empty = filename
        filenames.append(filename)

    if first_empty is not None:
        torch.save(state, first_empty)
    else:
        first = filenames[0]
        os.remove(first)
        for filename in filenames[1:]:
            os.rename(filename, first)
            first = filename
        torch.save(state, filenames[-1])


def build_checkpoint(
    exp_name,
    unique_id,
    tpe,
    model,
    optimizer,
    acc,
    loss,
    epoch,
    run_batches,
    is_preempted=False,
    start_batch=0,
):
    return {
        "exp_name": exp_name,
        "unique_id": unique_id,
        "type": tpe,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "acc": acc,
        "loss": loss,
        "epoch": epoch,
        "run_batches": run_batches,
        "is_preempted": is_preempted,
        "start_batch": start_batch,
    }


def restore_checkpoint(filename, model=None, optimizer=None):
    """restores checkpoint state from filename and load in model and optimizer if provided"""
    print(f"Extracting state from {filename}")
    if not os.path.exists(filename):
        print("No checkpoint file found")
        return None

    if torch.device == "cuda":
        state = torch.load(filename)
    else:
        state = torch.load(filename, map_location=torch.device("cpu"))

    if model:
        print(f"Loading model state_dict from state found in {filename}")
        model.load_state_dict(state["model"])
    if optimizer:
        print(f"Loading optimizer state_dict from state found in {filename}")
        optimizer.load_state_dict(state["optimizer"])
    return state


def restore_best_checkpoint_from_prefix(
    prefix, path="./checkpoints", model=None, optimizer=None
):
    filename = Path(path) / f"{prefix}_best"
    return restore_checkpoint(filename, model, optimizer)


def restore_best_checkpoint(
    exp_name,
    unique_id,
    tpe,
    model=None,
    optimizer=None,
    path="./checkpoints",
    extension="pth",
):
    filename = Path(path) / f"{exp_name}_{unique_id}_{tpe}_best.{extension}"
    return restore_checkpoint(filename, model, optimizer)
