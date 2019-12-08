# import checkpoints
# from tensorboard_utils import tensorboard_event_accumulator
from tensorboard_utils import Tensorboard
import utils
import model_process
import random
from math_dataset import (
    random_split_dataset,
    question_answer_to_position_batch_collate_fn,
    MathDatasetManager,
    FullDatasetManager,
)

# from transformer.Models import Transformer
import torch.optim as optim
from torch.utils import data
import torch

import numpy as np
import os

# import math
import multiprocessing

if __name__ == "__main__":
    # For laptop & deep learning rig testing on the same code
    if not torch.cuda.is_available():
        multiprocessing.set_start_method("spawn", True)
        device = torch.device("cpu")
        num_workers = 4
        max_elements = 2
    else:
        device = torch.device("cuda")
        num_workers = 16
        max_elements = None

    print("Device", device)

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

    print("Deterministic", deterministic)

    # mdsmgr = MathDatasetManager("./mathematics_dataset-v1.0")

    ds_train = FullDatasetManager(
        "./mathematics_dataset-v1.0", max_elements=max_elements
    )

    # print("types", list(mdsmgr.get_types()))
    # print("categories", list(mdsmgr.get_categories()))
    # print("modules: algebra", mdsmgr.get_modules_for_category("algebra"))

    exp_name = "math_easy_alge_l1d"
    unique_id = "11-24-19_1"

    # TODO: Figure out how to load the entire dataset

    # ds_train = mdsmgr.build_dataset_from_module(
    #     "algebra", "linear_1d", "train-easy", max_elements=max_elements
    # )
    # ds_interpolate = mdsmgr.build_dataset_from_module(
    #     "algebra", "linear_1d", "interpolate", max_elements=max_elements
    # )

    print("Train dataset size", len(ds_train))
    # print("Interpolate dataset size", len(ds_interpolate))

    model = utils.build_transformer()

    optimizer = optim.Adam(model.parameters(), lr=6e-6, betas=(0.9, 0.995), eps=1e-9)

    # here we split data in 90/10% for train/validation and use interpolate for test

    train_ds, val_ds = random_split_dataset(ds_train, split_rate=0.9)
    # we provide the function question_answer_to_position_batch_collate_fn that collates
    # all questions/answers into transformer format enhanced with char positioning
    train_loader = data.DataLoader(
        train_ds,
        batch_size=64,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=question_answer_to_position_batch_collate_fn,
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=question_answer_to_position_batch_collate_fn,
    )

    # interpolate_loader = data.DataLoader(
    #     ds_interpolate,
    #     batch_size=128,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     collate_fn=question_answer_to_position_batch_collate_fn,
    # )

    tb = Tensorboard(exp_name, unique_name=unique_id)

    model = model.to(device)

    model_process.train(
        exp_name,
        unique_id,
        model,
        train_loader,
        val_loader,
        # interpolate_loader,
        optimizer,
        device,
        epochs=5000,
        tb=tb,
        log_interval=100,
    )
