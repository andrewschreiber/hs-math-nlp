# import checkpoints
# from tensorboard_utils import tensorboard_event_accumulator
from tensorboard_utils import Tensorboard
from checkpoints import restore_checkpoint, load_latest_checkpoint_from_bucket
import utils
import time
import traceback
import model_process
import random

from math_dataset import (
    # random_split_dataset,
    question_answer_to_position_batch_collate_fn,
    # MathDatasetManager,
    FullDatasetManager,
)

# from transformer.Models import Transformer
import torch.optim as optim
from torch.utils import data
import torch
import torch.nn as nn

import numpy as np
import os

# import math
import multiprocessing
import signal


def main():
    print("Beginning training...")
    if utils.is_spot_instance():
        signal.signal(signal.SIGTERM, utils.sigterm_handler)
        print("Sigterm handler setup")

    # For laptop & deep learning rig testing on the same code
    if not torch.cuda.is_available():
        multiprocessing.set_start_method("spawn", True)
        device = torch.device("cpu")
        num_workers = 0
        max_elements = 5
        # max_batches = None
    else:
        device = torch.device("cuda")
        # num_workers = 16
        # https://github.com/facebookresearch/maskrcnn-benchmark/issues/195
        num_workers = 0

        # 666666 datapoints per difficulty file
        # 3 difficulties/module -> 2m datapoints/module
        # 56 modules
        # Total dataset of 112m
        # 224m rows (1 row per questions, 1 row per answer)

        # max_elements *per file*
        max_elements = None

        # Paper model trained for 500k batches with 1028 batch size
        #   = 512m datapoints used for training
        # 512m datapoints / 128 batch size = 4m batches
        # max_batches = 5000000

    print("Device:", device)

    # Paper calls for batch size of 1024
    # They use 8 P100s (16gb VRAM each) for 500k batches
    if torch.cuda.device_count() > 1:
        # Uses somewhere between 80-90GB VRAM
        batch_size = 1024
    else:
        # Uses ~10 GB VRAM
        # batch_size = 128
        batch_size = 32
    print("Batch size:", batch_size)

    start_epoch = 0
    print("Start epoch:", start_epoch)

    run_batches = 0

    should_restore_checkpoint = True
    print("Should restore checkpoint:", should_restore_checkpoint)

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

    print("Deterministic:", deterministic)

    exp_name = "math_112m_bs128"
    unique_id = "6-4-20_transformer_warmup"

    model = utils.build_transformer()

    lr = 6e-4
    beta_coeff_low = 0.9
    beta_coeff_high = 0.995
    eps = 1e-9

    print(
        f"Learning rate {lr}. B low {beta_coeff_low}. B high {beta_coeff_high}. eps{eps}"
    )

    optimizer = optim.Adam(
        model.parameters(), lr=lr, betas=(beta_coeff_low, beta_coeff_high), eps=eps
    )

    tb = Tensorboard(exp_name, unique_name=unique_id)
    start_batch = 0
    total_loss = 0
    n_char_total = 0
    n_char_correct = 0

    if should_restore_checkpoint:
        exp = f"{exp_name}_{unique_id}"
        # cp_path = f"checkpoints/{exp}_latest_checkpoint.pth"
        cp_path = "checkpoint_b109375_e0.pth"

        # state = load_latest_checkpoint_from_bucket(
        # exp=exp, model=model, optimizer=optimizer
        # )
        state = restore_checkpoint(cp_path, model=model, optimizer=optimizer,)

        if state is not None:
            start_epoch = state["epoch"]
            best_acc = state["acc"]
            best_loss = state["loss"]
            run_batches = state["run_batches"]

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

            print("start_epoch", start_epoch)
            print("start_batch", start_batch)
            print("best_acc", best_acc)
            print("best_loss", best_loss)
            print("Loaded checkpoint successfully")

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    dataset_path = "./mathematics_dataset-v1.0"

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

    run_max_batches = 500000  # Defined in paper

    # og_datapoint_iterations = 500000 * 1024  # Paper Batches * batch_size

    # run_max_batches = og_datapoint_iterations / batch_size - 2 * 875000  # 2 epochs

    # print(f"Calculated max batches: {run_max_batches}")

    # we provide the function question_answer_to_position_batch_collate_fn that collates
    # all questions/answers into transformer format enhanced with char positioning

    train_loader = data.DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=question_answer_to_position_batch_collate_fn,
        pin_memory=True,
    )

    interpolate_loader = data.DataLoader(
        ds_interpolate,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=question_answer_to_position_batch_collate_fn,
        pin_memory=True,
    )

    extrapolate_loader = data.DataLoader(
        ds_extrapolate,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=question_answer_to_position_batch_collate_fn,
        pin_memory=True,
    )

    model_process.train(
        name=f"{exp_name}_{unique_id}",
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
        checkpoint=True,  # Only save on GPUs
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
