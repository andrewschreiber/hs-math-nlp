import torch
import time
import numpy as np
import os

from math_dataset import (
    VOCAB_SZ,
    MAX_QUESTION_SZ,
    MAX_ANSWER_SZ,
    lstm_batch_collate_fn,
    question_answer_to_position_batch_collate_fn,
)
from transformer.Models import Transformer
from LSTM.simple import SimpleLSTM
from training import TRANSFORMER, SIMPLE_LSTM, ATTENTIONAL_LSTM


def one_hot_seq(chars, vocab_size=VOCAB_SZ, char0=ord(" ")):
    chars = (chars - char0).long()
    return torch.zeros(len(chars), VOCAB_SZ + 1).scatter_(1, chars.unsqueeze(1), 1.0)


def torch_one_hot_encode_string(s):
    chars = np.array(list(s), dtype="S1").view(np.uint8)
    q = torch.tensor(chars, dtype=torch.uint8)
    q = one_hot_seq(q)
    return q


def collate_fn(model_type):
    if model_type == TRANSFORMER:
        return question_answer_to_position_batch_collate_fn
    elif model_type == SIMPLE_LSTM:
        return lstm_batch_collate_fn
    elif model_type == ATTENTIONAL_LSTM:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid model_type {model_type}.")


def build_model(model_type, weight_sharing):
    if model_type == TRANSFORMER:
        return build_transformer(weight_sharing=weight_sharing)
    elif model_type == SIMPLE_LSTM:
        return build_simple_lstm()
    elif model_type == ATTENTIONAL_LSTM:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid model_type {model_type}.")


def build_transformer(
    n_src_vocab=VOCAB_SZ + 1,
    n_tgt_vocab=VOCAB_SZ + 1,
    len_max_seq_encoder=MAX_QUESTION_SZ,
    len_max_seq_decoder=MAX_ANSWER_SZ,
    built_in=False,
    weight_sharing=True,
):
    if built_in:
        raise NotImplementedError("Fix input shape error")
        return torch.nn.Transformer()

    return Transformer(
        n_src_vocab=n_src_vocab,  # add PAD in vocabulary
        n_tgt_vocab=n_tgt_vocab,  # add PAD in vocabulary
        len_max_seq_encoder=len_max_seq_encoder,
        len_max_seq_decoder=len_max_seq_decoder,
        tgt_emb_prj_weight_sharing=weight_sharing,
        emb_src_tgt_weight_sharing=weight_sharing,
    )


def build_simple_lstm():
    return SimpleLSTM(VOCAB_SZ, MAX_ANSWER_SZ, MAX_QUESTION_SZ)


def build_att_lstm():
    return None


def is_preempted():
    return os.environ.get("IS_PREEMPTED", None) == "TRUE"


def sigterm_handler(sig, frame):
    print("Got SIGTERM. Setting `IS_PREEMPTED` to true.")
    os.environ["IS_PREEMPTED"] = "TRUE"


def is_spot_instance():
    # TODO: Find os.environ flag / metadata request to check
    return True


def is_cloud():
    from sys import platform

    if platform == "linux" or platform == "linux2":
        return True
    elif platform == "darwin" or "win32":
        return False
    else:
        print(f"Unknown platform {platform}. Assuming is_cloud == True")
        return True


def shutdown():
    if is_cloud():
        print("Shutting down in 60 seconds...")
        time.sleep(60)
        print(f"Shutting down at {time.time()}")
        os.system("sudo shutdown -h now")
