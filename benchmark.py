import torch
import numpy as np
import checkpoints

import model_process
import utils
import matplotlib.pyplot as plt
from torch.utils import data
from loss import compute_performance

from transformer import Constants
from transformer.Generator import Generator
from math_dataset import (
    MathDatasetManager,
    LazyFileMathDataset,
    BenchmarkDatasetManager,
    benchmark_collate_fn,
    MAX_QUESTION_SZ,
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
generator = Generator(
    model, device, beam_size=5, max_token_seq_len=MAX_QUESTION_SZ, n_best=1,
)
results = {}
for module, dataset in benchmark.get_datasets("interpolate").items():
    print(f"Testing {module} of length {len(dataset)} ...")
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=benchmark_collate_fn,
        pin_memory=False,
    )
    iterator = iter(loader)

    resps = model_process.predict_benchmark(generator, iterator, device)
    correct = 0
    for resp in resps:
        if resp["correct"] is True:
            correct += 1

    print(f"Got {correct} of {len(dataset)} correct in {module}.")
    results[module] = correct

print(results)
print("Benchmark complete")
