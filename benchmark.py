import torch
import time
import checkpoints

import model_process
import utils
from torch.utils import data
import torch.nn as nn

from transformer.Generator import Generator
from math_dataset import (
    BenchmarkDatasetManager,
    benchmark_collate_fn,
    MAX_QUESTION_SZ,
)

WORKSPACE_FOLDER = "/Users/andrew/git/hs-math-nlp"

print(f"Running benchmarks at {time.time()}")
model_filepath = f"{WORKSPACE_FOLDER}/checkpoints/checkpoint_b500000_e4_complete.pth"

# build default transformer model
model = utils.build_transformer()
# restore model from checkpoint
_ = checkpoints.restore_checkpoint(model_filepath, model)

if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
print("device", device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

ds_path = f"{WORKSPACE_FOLDER}/mathematics_dataset-v1.0"
benchmark = BenchmarkDatasetManager(ds_path)

batch_size = 128
generator = Generator(
    model, device, beam_size=5, max_token_seq_len=MAX_QUESTION_SZ, n_best=1,
)
results = {}
for module, dataset in benchmark.get_datasets("interpolate").items():
    print(f"Testing {module} of length {len(dataset)} ...")
    start = time.time()
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

    print(f"S: {(time.time() - start) * 1000}ms")
    print(
        f"Got {correct} of {len(dataset)} correct in {module} after {(time.time() - start)}s."
    )
    results[module] = correct

print(results)
print("Benchmark complete")
