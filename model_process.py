import time
import math
from tqdm import tqdm  # tqdm_notebook as tqdm
import numpy as np
import os
import torch
from torch.utils import data
import utils
import sys

# import torch.nn.functional as F
from transformer import Constants
from transformer.Generator import Generator

# from math_dataset import VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ, np_decode_string
from math_dataset import MAX_QUESTION_SZ, np_decode_string
from loss import compute_performance
from checkpoints import (
    rotating_save_checkpoint,
    build_checkpoint,
    save_checkpoint_to_bucket,
)
from math_dataset import np_encode_string, question_to_position_batch_collate_fn


def train(
    exp_name,
    unique_id,
    model,
    training_data,
    optimizer,
    device,
    epochs,
    validation_data=None,
    tb=None,
    log_interval=100,
    interpolate_interval=1,
    interpolate_data=None,
    start_epoch=0,
    start_batch=0,
    run_batches=0,
    best_valid_accu=0.0,
    best_valid_loss=float("Inf"),
    best_interpolate_accu=0.0,
    best_interpolate_loss=float("Inf"),
    run_max_batches=None,
    extrapolate_data=None,
    checkpoint=True,
):
    print("~~~ Beginning Training ~~~~")
    print(
        f"Start epoch: {start_epoch}, Start batch: {start_batch}, Max batch: {run_max_batches}"
    )

    for epoch_i in range(start_epoch, epochs):
        print(
            f"[ Epoch: {epoch_i + 1} / {epochs}, Run Batch: {start_batch + run_batches} / {run_max_batches}]"
        )
        epoch_max_batches_remaining = (
            run_max_batches - run_batches if run_max_batches is not None else None
        )

        start = time.time()
        train_loss, train_accu, new_batch_count, interrupted_batch, done = train_epoch(
            model=model,
            training_data=training_data,
            optimizer=optimizer,
            device=device,
            epoch=epoch_i,
            tb=tb,
            log_interval=log_interval,
            max_batches=epoch_max_batches_remaining,
            run_batch_count=run_batches,
            start_batch=start_batch,
        )
        start_batch = 0
        run_batches = new_batch_count

        print(
            "[Training]  loss: {train_loss}, ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, "
            "elapse: {elapse:3.3f}ms".format(
                train_loss=train_loss,
                ppl=math.exp(min(train_loss, 100)),
                accu=100 * train_accu,
                elapse=(time.time() - start) * 1000,
            )
        )

        if not utils.is_preempted():
            inference_datasets = {}
            if interpolate_data:
                inference_datasets["interpolate"] = interpolate_data
            if extrapolate_data:
                inference_datasets["extrapolate"] = extrapolate_data

            for group, dataset in inference_datasets.items():
                start = time.time()
                inference_loss, inference_acc = inference_epoch(
                    model, dataset, device, epoch_i, group, tb, log_interval,
                )
                print(
                    "[{group}]  loss: {inference_loss},  ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, "
                    "elapse: {elapse:3.3f}ms".format(
                        group=group,
                        inference_loss=inference_loss,
                        ppl=math.exp(min(inference_loss, 100)),
                        accu=100 * inference_acc,
                        elapse=(time.time() - start) * 1000,
                    )
                )

        if utils.is_preempted():  # TODO: or interrupted_batch is not None
            # TODO: Or checkpoint. Must add different naming logic.
            print("Building checkpoint..")
            start = time.time()
            state = build_checkpoint(
                exp_name=exp_name,
                unique_id=unique_id,
                tpe="training",
                model=model,
                optimizer=optimizer,
                acc=train_accu,
                loss=train_loss,
                epoch=epoch_i,
                run_batches=run_batches,
                is_preempted=utils.is_preempted(),
                start_batch=interrupted_batch + 1
                if interrupted_batch is not None
                else 0,
            )

            if utils.is_cloud():
                print("Saving to google cloud..")
                save_checkpoint_to_bucket(
                    state=state,
                    preempted=utils.is_preempted(),
                    exp=f"{exp_name}_{unique_id}",
                    path="./checkpoints",
                )
            else:
                rotating_save_checkpoint(
                    state,
                    prefix=f"{exp_name}_{unique_id}_{run_batches}_training",
                    path="./checkpoints",
                    nb=5,
                )
            print(f"Save checkpoint time: {(time.time() - start) * 1000}")
            if utils.is_preempted():
                print("Completed preemption handling. Cleanly exiting.")
                sys.exit(0)

        if done:
            print(
                f"Reached max batch. Breaking out of training at the end of epoch {epoch_i}"
            )
            break
        training_data.dataset.shuffleData()

    print("~~~~~~ Completed training ~~~~~~")

    if utils.is_cloud():
        print("Shutting down instance")
        os.system("sudo shutdown -h now")


def train_epoch(
    model,
    training_data,
    optimizer,
    device,
    epoch,
    tb=None,
    log_interval=100,
    max_batches=None,
    run_batch_count=0,
    start_batch=0,
):
    if start_batch != 0:
        print(f"Training within batch {start_batch}")

    model.train()
    total_loss = 0
    n_char_total = 0
    n_char_correct = 0
    interrupted_batch = None
    done = False

    for batch_idx, batch in enumerate(
        tqdm(
            training_data,
            mininterval=2,
            leave=False,
            disable=utils.is_cloud(),
            dynamic_ncols=not utils.is_cloud(),
        ),
        start=start_batch,
    ):
        batch_qs, batch_qs_pos, batch_as, batch_as_pos = map(
            lambda x: x.to(device), batch
        )
        gold_as = batch_as[:, 1:]

        optimizer.zero_grad()

        pred_as = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos)

        loss, n_correct = compute_performance(pred_as, gold_as, smoothing=True)
        loss.backward()

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold_as.ne(Constants.PAD)
        n_char = non_pad_mask.sum().item()
        n_char_total += n_char
        n_char_correct += n_correct

        if tb is not None and batch_idx % log_interval == 0:
            tb.add_scalars(
                {
                    "loss_per_char": total_loss / n_char_total,
                    "accuracy": n_char_correct / n_char_total,
                },
                group="train",
                sub_group="batch",
                global_step=epoch * len(training_data) + batch_idx,
            )

        if max_batches is not None and run_batch_count == max_batches:
            print(
                f"Reached {run_batch_count} batches on max_batches of {max_batches}. Breaking from epoch."
            )
            # interrupted_batch = batch_idx
            done = True
            break
        if utils.is_preempted():
            print(
                f"Preemption at end of Epoch batch: {batch_idx} and Run batch: {run_batch_count}. Breaking from epoch."
            )
            interrupted_batch = batch_idx
            break
        run_batch_count += 1

    loss_per_char = total_loss / n_char_total
    accuracy = n_char_correct / n_char_total

    if tb is not None:
        tb.add_scalars(
            {"loss_per_char": loss_per_char, "accuracy": accuracy},
            group="train",
            sub_group="epoch",
            global_step=epoch,
        )

    return loss_per_char, accuracy, run_batch_count, interrupted_batch, done


def inference_epoch(model, data, device, epoch, group, tb=None, log_interval=100):
    model.eval()

    total_loss = 0
    n_char_total = 0
    n_char_correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data, mininterval=2, leave=False)):
            # prepare data
            batch_qs, batch_qs_pos, batch_as, batch_as_pos = map(
                lambda x: x.to(device), batch
            )
            gold_as = batch_as[:, 1:]

            # forward
            pred_as = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos)
            loss, n_correct = compute_performance(pred_as, gold_as, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold_as.ne(Constants.PAD)
            n_char = non_pad_mask.sum().item()
            n_char_total += n_char
            n_char_correct += n_correct

    loss_per_char = total_loss / n_char_total
    accuracy = n_char_correct / n_char_total

    if tb is not None:
        tb.add_scalars(
            {"loss_per_char": loss_per_char, "accuracy": accuracy},
            group=group,
            sub_group="epoch",
            global_step=epoch,
        )

    return loss_per_char, accuracy


def predict(generator, data, device, max_predictions=None):
    if max_predictions is not None:
        cur = max_predictions
    else:
        cur = len(data)

    resps = []
    for batch_idx, batch in enumerate(data):
        if cur == 0:
            break

        batch_qs, batch_qs_pos = map(lambda x: x.to(device), batch)
        all_hyp, all_scores = generator.generate_batch(batch_qs, batch_qs_pos)

        for i, idx_seqs in enumerate(all_hyp):
            for j, idx_seq in enumerate(idx_seqs):
                r = np_decode_string(np.array(idx_seq))
                s = all_scores[i][j].cpu().item()
                resps.append({"resp": r, "score": s})
        cur -= 1

    return resps


def predict_dataset(
    dataset,
    model,
    device,
    callback,
    max_batches=None,
    beam_size=5,
    max_token_seq_len=MAX_QUESTION_SZ,
    n_best=1,
    batch_size=1,
    num_workers=1,
):

    generator = Generator(
        model,
        device,
        beam_size=beam_size,
        max_token_seq_len=max_token_seq_len,
        n_best=n_best,
    )

    if max_batches is not None:
        cur = max_batches
    else:
        cur = len(dataset)

    resps = []
    for batch_idx, batch in enumerate(dataset):
        if cur == 0:
            break

        batch_qs, batch_qs_pos, _, _ = map(lambda x: x.to(device), batch)
        all_hyp, all_scores = generator.generate_batch(batch_qs, batch_qs_pos)

        callback(batch_idx, all_hyp, all_scores)

        cur -= 1
    return resps


def predict_multiple(
    questions,
    model,
    device,
    beam_size=5,
    max_token_seq_len=MAX_QUESTION_SZ,
    n_best=1,
    batch_size=1,
    num_workers=1,
):

    questions = list(map(lambda q: np_encode_string(q), questions))
    questions = data.DataLoader(
        questions,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=question_to_position_batch_collate_fn,
    )

    generator = Generator(
        model,
        device,
        beam_size=beam_size,
        max_token_seq_len=max_token_seq_len,
        n_best=n_best,
    )

    return predict(generator, questions, device)


def predict_single(
    question, model, device, beam_size=5, max_token_seq_len=MAX_QUESTION_SZ, n_best=1
):

    generator = Generator(
        model,
        device,
        beam_size=beam_size,
        max_token_seq_len=max_token_seq_len,
        n_best=n_best,
    )

    qs = [np_encode_string(question)]
    qs, qs_pos = question_to_position_batch_collate_fn(qs)
    qs, qs_pos = qs.to(device), qs_pos.to(device)

    all_hyp, all_scores = generator.generate_batch(qs, qs_pos)
    # resp = np_decode_string(np.array(all_hyp[0][0]))

    resps = []
    for i, idx_seqs in enumerate(all_hyp):
        for j, idx_seq in enumerate(idx_seqs):
            r = np_decode_string(np.array(idx_seq))
            s = all_scores[i][j].cpu().item()
            resps.append({"resp": r, "score": s})

    return resps
