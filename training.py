import checkpoints
from tensorboard_utils import tensorboard_event_accumulator
from tensorboard_utils import Tensorboard
import utils
import model_process
from math_dataset import (
    random_split_dataset,
    question_answer_to_position_batch_collate_fn
)
from transformer.Models import Transformer
from math_dataset import MathDatasetManager
import math_dataset
import torch.optim as optim
from torch.utils import data
import torch
import numpy as np
import math
import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', True)
    mdsmgr = MathDatasetManager(
        "./mathematics_dataset-v1.0"
    )

    print("types", list(mdsmgr.get_types()))
    print("categories", list(mdsmgr.get_categories()))
    print("modules of arithmetic", mdsmgr.get_modules_for_category('arithmetic'))

    # ds = ""

    #ds = mdsmgr.build_dataset_from_module('arithmetic', 'add_or_sub', 'train-easy')
    # print("size", len(ds))

    # ds = mdsmgr.build_dataset_from_module(
    #    'arithmetic', 'add_or_sub', 'train-easy', max_elements=1000)
    # print("size", len(ds))

    # ds = mdsmgr.build_dataset_from_modules(
    #    'arithmetic', ['add_or_sub', 'add_sub_multiple'], 'train-easy')
    # print("size", len(ds))

    # ds = mdsmgr.build_dataset_from_category('arithmetic', 'train-easy')
    # print("size", len(ds))

    # ds = mdsmgr.build_dataset_from_categories(
    #    ['arithmetic', 'polynomials'], 'train-easy')
    # print("size", len(ds))


    seed = 1
    torch.manual_seed(seed)
    # device = torch.device("cuda")
    device = torch.device("cpu")
    print("device", device)
    # torch.cuda.synchronize()

    exp_name = "math_easy_arith"
    unique_id = "11-24-19_1"

    # ds = mdsmgr.build_dataset_from_module(
    #    'algebra', 'linear_1d', 'train-easy')
    # print("train-easy dataset size", len(ds))

    # ds_interpolate = mdsmgr.build_dataset_from_module(
        # 'algebra', 'linear_1d', 'interpolate'
    # )

    # print("interpolate dataset size", len(ds_interpolate))


    # ds_train = mdsmgr.build_dataset_from_categories(
        # ['arithmetic', 'polynomials'], 'train-easy')

    ds_train = mdsmgr.build_dataset_from_category(
        'arithmetic', 'train-easy', max_elements=1000
    )
    ds_interpolate = mdsmgr.build_dataset_from_category(
        'arithmetic', 'interpolate', max_elements=1000
    )

    # ds_train = mdsmgr.build_dataset_from_level('train-easy')

    print("Train dataset size", len(ds_train))

    model = utils.build_transformer()

    optimizer = optim.Adam(model.parameters(), lr=6e-6,
                        betas=(0.9, 0.995), eps=1e-9)

    # here we split data in 90/10% for train/validation and use interpolate for test
    train_ds, val_ds = math_dataset.random_split_dataset(ds_train, split_rate=0.9)

    # we provide the function question_answer_to_position_batch_collate_fn that collates
    # all questions/answers into transformer format enhanced with char positioning
    train_loader = data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=4,
        collate_fn=question_answer_to_position_batch_collate_fn)

    val_loader = data.DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=4,
        collate_fn=question_answer_to_position_batch_collate_fn)

    interpolate_loader = data.DataLoader(
        ds_interpolate, batch_size=128, shuffle=False, num_workers=4,
        collate_fn=question_answer_to_position_batch_collate_fn)

    tb = Tensorboard(exp_name, unique_name=unique_id)

    model = model.to(device)

    model_process.train(
        exp_name, unique_id,
        model,
        train_loader, val_loader, interpolate_loader,
        optimizer, device,
        epochs=5000, tb=tb, log_interval=100,
    )
