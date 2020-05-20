import torch
import numpy as np
import checkpoints

import model_process
import utils
import matplotlib.pyplot as plt
from torch.utils import data
from loss import compute_performance

from transformer import Constants
from math_dataset import (
    MathDatasetManager,
    LazyFileMathDataset,
    BenchmarkDatasetManager,
    benchmark_collate_fn,
)

device = torch.device("cpu")
print("device", device)

model_filepath = (
    "/Users/andrew/git/hs-math-nlp/checkpoints/checkpoint_b500000_e4_complete.pth"
)
# build default transformer model
model = utils.build_transformer()
# restore model from checkpoint
_ = checkpoints.restore_checkpoint(model_filepath, model)
# mdsmgr = MathDatasetManager("/Users/andrew/git/hs-math-nlp/mathematics_dataset-v1.0")

ds_path = "/Users/andrew/git/hs-math-nlp/mathematics_dataset-v1.0"
benchmark = BenchmarkDatasetManager(ds_path)


# ds = mdsmgr.build_dataset_from_file(f"{ds_path}/{typ}/{file}")
# lz = LazyFileMathDataset(filepath)
batch_size = 128

for module, dataset in benchmark.get_datasets("interpolate").items():
    print(f"Testing {module} ...")
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=benchmark_collate_fn,
        pin_memory=False,
    )
    iterator = iter(loader)
    batch_qs, batch_qs_pos, batch_string_as = next(iterator)

    print(batch_qs, batch_qs_pos, batch_string_as)


print("Benchmark complete")
