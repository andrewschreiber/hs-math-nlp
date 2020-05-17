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
    question_answer_to_position_batch_collate_fn,
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
        collate_fn=question_answer_to_position_batch_collate_fn,
        pin_memory=False,
    )
    iterator = iter(loader)

    total_loss = 0
    n_char_total = 0
    n_char_correct = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            # prepare data
            batch_qs, batch_qs_pos, batch_as, batch_as_pos = map(
                lambda x: x.to(device), batch
            )
            gold_as = batch_as[:, 1:]

            pred_as = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos)
            loss, n_correct = compute_performance(pred_as, gold_as, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold_as.ne(Constants.PAD)
            n_char = non_pad_mask.sum().item()
            n_char_total += n_char
            n_char_correct += n_correct

            print(f"{n_correct} / {n_char} correct on {batch_idx}")

    loss_per_char = total_loss / n_char_total
    accuracy = n_char_correct / n_char_total
    print(f"{module} complete. Acc: {accuracy}. Loss_per_char: {loss_per_char}")

print("Done.")
