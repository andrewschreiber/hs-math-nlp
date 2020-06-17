import os
from pathlib import Path
from google.cloud import storage
import torch
import utils

BUCKET_NAME = "math-checkpoints-data"

# TODO: Refactor naming of checkpointing functions
# Currently just saves locally


def save_checkpoint(state, name, path):
    if not os.path.isdir(path):
        os.makedirs(path)

    filename = f"{name}.pth"
    filepath = Path(path) / filename

    if utils.is_preempted():
        print("Preempted, skipping model save")
        return

    try:
        print(f"Removing existing model file at {filepath}")
        os.remove(filepath)
        # checkpoint_count = len(os.listdir(os.getcwd()))
        # new_name = f"{checkpoint_count}_{filename}"
        # os.rename(filepath, new_name)
        # print(f"Found existing model, renaming to {new_name}")
    except OSError:
        print("No existing model file found")
        pass

    print(f"Starting checkpoint save of {filepath}...")
    torch.save(state, filepath)
    print(f"Final saved model size: {os.stat(filepath).st_size}")

    # Disable bucket saving for now due to torch.save 0 byte error

    # storage_client = storage.Client()
    # bucket = storage_client.bucket(BUCKET_NAME)
    # blob = bucket.blob(filename)

    # print("Started bucket upload...")
    # blob.upload_from_filename(filepath)
    # print(f"File {filename} uploaded to {BUCKET_NAME}.")


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
    print(f"Found checkpoint in bucket: {BUCKET_NAME}")
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
    name,
    model,
    optimizer,
    acc,
    loss,
    epoch,
    run_batches,
    lr,
    tpe="training",
    is_preempted=False,
    start_batch=0,
    total_loss=0,
    n_char_total=0,
    n_char_correct=0,
):
    return {
        "name": name,
        "type": tpe,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "acc": acc,
        "loss": loss,
        "epoch": epoch,
        "run_batches": run_batches,
        "is_preempted": is_preempted,
        "start_batch": start_batch,
        "total_loss": total_loss,
        "n_char_total": n_char_total,
        "n_char_correct": n_char_correct,
        "lr": lr,
    }


def restore_checkpoint(filename, model_type, model=None, optimizer=None):
    """restores checkpoint state from filename and load in model and optimizer if provided"""
    print(f"Attempting to extract state from {filename}...")
    if not os.path.exists(filename):
        print("No checkpoint file found")
        return None

    if torch.device == "cuda":
        state = torch.load(filename)
    else:
        state = torch.load(filename, map_location=torch.device("cpu"))

    if model:
        print(f"Loading model state_dict from state found in {filename}")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state["model"].items():
            name = k[7:] if "module." in k else k  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        # model.load_state_dict(state["model"])
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
