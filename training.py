import time
import traceback
import random
import torch
import torch.optim as optim
from torch.utils import data
import numpy as np
import os
import multiprocessing
import signal
import argparse
from pathlib import Path

import utils
from math_dataset import FullDatasetManager
import model_process
from tensorboard_utils import Tensorboard
from checkpoints import restore_checkpoint

TRANSFORMER = "transformer"
SIMPLE_LSTM = "simLSTM"
ATTENTIONAL_LSTM = "attLSTM"
MODELS = [TRANSFORMER, SIMPLE_LSTM, ATTENTIONAL_LSTM]

# Run with:
#   python training.py -m simLSTM
# OR, change the default argument in parser:
#   python training.py


def main():
    print(f"Beginning training at {time.time()}...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="Name of the model", choices=MODELS, default=TRANSFORMER,
    )
    args = parser.parse_args()
    model_type = args.model

    if utils.is_spot_instance():
        signal.signal(signal.SIGTERM, utils.sigterm_handler)

    # For laptop & deep learning rig testing on the same codebase
    if not torch.cuda.is_available():
        multiprocessing.set_start_method("spawn", True)
        device = torch.device("cpu")
        num_workers = 0
        max_elements = 5
        save_checkpoints = False
    else:
        device = torch.device("cuda")
        # https://github.com/facebookresearch/maskrcnn-benchmark/issues/195
        num_workers = 0
        max_elements = None
        save_checkpoints = True

    deterministic = True
    if deterministic:
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    shuffle = not deterministic

    # Hyperparameters
    batch_size = 1024 if torch.cuda.device_count() > 1 else 8
    lr = 6e-4
    warmup_lr = 6e-6  # TODO: Refactor into custom optimizer class
    warmup_interval = None  # 10000  # or None
    beta_coeff_low = 0.9
    beta_coeff_high = 0.995
    eps = 1e-9
    smoothing = False
    weight_sharing = True

    # Config
    unique_id = f"6-24-20_{model_type}1"
    exp = "math_112m_bs128"
    name = f"{exp}_{unique_id}"
    run_max_batches = 500000  # Defined in paper
    should_restore_checkpoint = True
    pin_memory = True

    print("Model name:", name)
    print(
        f"Batch size: {batch_size}. Learning rate: {lr}. Warmup_lr: {warmup_lr}. Warmup interval: {warmup_interval}. B low {beta_coeff_low}. B high {beta_coeff_high}. eps {eps}. Smooth: {smoothing}"
    )
    print("Deterministic:", deterministic)
    print("Device:", device)
    print("Should restore checkpoint:", should_restore_checkpoint)

    model = utils.build_model(model_type, weight_sharing)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr if warmup_interval is None else warmup_lr,
        betas=(beta_coeff_low, beta_coeff_high),
        eps=eps,
    )

    tb = Tensorboard(exp, unique_name=unique_id)

    # Run state
    start_batch = 0
    start_epoch = 0
    run_batches = 0
    total_loss = 0
    n_char_total = 0
    n_char_correct = 0

    if should_restore_checkpoint:
        cp_path = f"checkpoints/{name}_latest_checkpoint.pth"
        # cp_path = "checkpoint_b109375_e0.pth"

        state = restore_checkpoint(
            cp_path, model_type=model_type, model=model, optimizer=optimizer,
        )

        if state is not None:
            start_epoch = state["epoch"]
            # best_acc = state["acc"]
            # best_loss = state["loss"]
            run_batches = state["run_batches"]
            lr = state["lr"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            start_batch = state.get("start_batch", None) or 0
            total_loss = state.get("total_loss", None) or 0
            n_char_total = state.get("n_char_total", None) or 0
            n_char_correct = state.get("n_char_correct", None) or 0

            # Need to move optimizer state to GPU memory
            if torch.cuda.is_available():
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

            print(f"Setting lr to {lr}")
            print("Loaded checkpoint successfully")

    print("start_epoch", start_epoch)
    print("start_batch", start_batch)
    print("total_loss", total_loss)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    dataset_path = "./mathematics_dataset-v1.0"
    mini_dataset_path = "./mini_dataset"
    if not os.path.isdir(Path(dataset_path)):
        print("Full dataset not detected. Using backup mini dataset for testing. See repo for instructions on downloading full dataset.")
        dataset_path = mini_dataset_path

    ds_train = FullDatasetManager(
        dataset_path,
        max_elements=max_elements,
        deterministic=deterministic,
        start_epoch=start_epoch,
        start_datapoint=start_batch * batch_size,
    )
    print("Train size:", len(ds_train))

    ds_interpolate = FullDatasetManager(
        dataset_path,
        max_elements=max_elements,
        deterministic=deterministic,
        start_epoch=start_epoch,
        mode="interpolate",
    )
    print("Interpolate size:", len(ds_interpolate))

    ds_extrapolate = FullDatasetManager(
        dataset_path,
        max_elements=max_elements,
        deterministic=deterministic,
        start_epoch=start_epoch,
        mode="extrapolate",
    )
    print("Extrapolate size:", len(ds_extrapolate))

    collate_fn = utils.collate_fn(model_type)

    train_loader = data.DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    interpolate_loader = data.DataLoader(
        ds_interpolate,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    extrapolate_loader = data.DataLoader(
        ds_extrapolate,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    model_process.train(
        name=name,
        model=model,
        training_data=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=1000,  # Not relevant, will get ended before this due to max_b
        tb=tb,
        run_max_batches=run_max_batches,
        validation_data=None,
        start_epoch=start_epoch,
        start_batch=start_batch,
        total_loss=total_loss,
        n_char_total=n_char_total,
        n_char_correct=n_char_correct,
        run_batches=run_batches,
        interpolate_data=interpolate_loader,
        extrapolate_data=extrapolate_loader,
        checkpoint=save_checkpoints,
        lr=lr,
        warmup_lr=warmup_lr,
        warmup_interval=warmup_interval,
        smoothing=smoothing,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ending script")
    except BaseException:
        print("Catching error...")
        print(traceback.format_exc())
        utils.shutdown()
        raise
