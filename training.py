# import checkpoints
# from tensorboard_utils import tensorboard_event_accumulator
from tensorboard_utils import Tensorboard
from checkpoints import restore_checkpoint
import utils
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


def sigterm_handler(sig, frame):
    print("Got SIGTERM. Setting `IS_PREEMPTED` to true.")
    os.environ["IS_PREEMPTED"] = "TRUE"


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, sigterm_handler)
    # For laptop & deep learning rig testing on the same code
    if not torch.cuda.is_available():
        multiprocessing.set_start_method("spawn", True)
        device = torch.device("cpu")
        num_workers = 0
        max_elements = 2
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
        batch_size = 128
    print("Batch size:", batch_size)

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

    # mdsmgr = MathDatasetManager("./mathematics_dataset-v1.0")

    ds_train = FullDatasetManager(
        "./mathematics_dataset-v1.0", max_elements=max_elements
    )
    # one of the options here is to type comments so fast that she has to listen and so on and so forth.
    #

    # print("types", list(mdsmgr.get_types()))
    # print("categories", list(mdsmgr.get_categories()))
    # print("modules: algebra", mdsmgr.get_modules_for_category("algebra"))

    exp_name = "math_112m_bs128"
    unique_id = "1-24-20_final"

    # ds_train = mdsmgr.build_dataset_from_module(
    #     "algebra", "linear_1d", "train-easy", max_elements=max_elements
    # )

    # TODO: Use full interpolate dataset
    # ds_interpolate = mdsmgr.build_dataset_from_module(
    #     "algebra", "linear_1d", "interpolate", max_elements=max_elements
    # )

    print("Train dataset size", len(ds_train))
    # print("Interpolate dataset size", len(ds_interpolate))

    model = utils.build_transformer()

    optimizer = optim.Adam(model.parameters(), lr=6e-6, betas=(0.9, 0.995), eps=1e-9)

    # val_loader = data.DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     collate_fn=question_answer_to_position_batch_collate_fn,
    # )

    # interpolate_loader = data.DataLoader(
    #     ds_interpolate,
    #     batch_size=128,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     collate_fn=question_answer_to_position_batch_collate_fn,
    # )

    tb = Tensorboard(exp_name, unique_name=unique_id)

    state = restore_checkpoint(
        "checkpoints/math_112m_bs128_1-12-20_1training_best.pth",
        model=model,
        optimizer=optimizer,
    )
    epoch = state["epoch"]
    best_acc = state["acc"]
    best_loss = state["loss"]

    start_batch = state.get("start_batch", None) or 0

    # print("exp_name", exp_name)
    # print("unique_id", unique_id)
    # print("epoch", epoch)
    # print("best_acc", best_acc)
    # print("best_loss", best_loss)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    og_datapoint_iterations = 500000 * 1024  # Paper Batches * batch_size

    run_max_batches = og_datapoint_iterations / batch_size - 875000  # 1 epoch in

    print(f"Calculated max batches: {run_max_batches}")

    # we provide the function question_answer_to_position_batch_collate_fn that collates
    # all questions/answers into transformer format enhanced with char positioning

    # train_ds = ds_train[start_batch:]
    train_ds = ds_train

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=question_answer_to_position_batch_collate_fn,
    )

    model_process.train(
        exp_name=exp_name,
        unique_id=unique_id,
        model=model,
        training_data=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=99,  # Not relevant, will get ended before this due to max_b
        tb=tb,
        run_max_batches=run_max_batches,
        validation_data=None,
        start_epoch=1,
        start_batch=start_batch,
    )

    # model_process.train(
    #     exp_name,
    #     unique_id,
    #     model,
    #     train_loader,
    #     optimizer,
    #     device,
    #     epochs=5000,
    #     tb=tb,
    #     log_interval=100,
    #     max_batches=max_batches,
    # )
