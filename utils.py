import torch
import numpy as np
import os
from math_dataset import VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ
from transformer.Models import Transformer


def one_hot_seq(chars, vocab_size=VOCAB_SZ, char0=ord(" ")):
    chars = (chars - char0).long()
    return torch.zeros(len(chars), VOCAB_SZ + 1).scatter_(1, chars.unsqueeze(1), 1.0)


def torch_one_hot_encode_string(s):
    chars = np.array(list(s), dtype="S1").view(np.uint8)
    q = torch.tensor(chars, dtype=torch.uint8)
    q = one_hot_seq(q)
    return q


def build_transformer(
    n_src_vocab=VOCAB_SZ + 1,
    n_tgt_vocab=VOCAB_SZ + 1,
    len_max_seq_encoder=MAX_QUESTION_SZ,
    len_max_seq_decoder=MAX_ANSWER_SZ,
):
    return Transformer(
        n_src_vocab=n_src_vocab,  # add PAD in vocabulary
        n_tgt_vocab=n_tgt_vocab,  # add PAD in vocabulary
        len_max_seq_encoder=len_max_seq_encoder,
        len_max_seq_decoder=len_max_seq_decoder,
    )


def is_preempted():
    return os.environ.get("IS_PREEMPTED", None) == "TRUE"


def sigterm_handler(sig, frame):
    print("Got SIGTERM. Setting `IS_PREEMPTED` to true.")
    f = open("testing_preemption.txt", "w+")
    for i in range(10):
        f.write("This is line %d\r\n" % (i + 1))
    f.close()

    os.environ["IS_PREEMPTED"] = "TRUE"


def is_spot_instance():
    # TODO: Find os.environ flag / metadata request to check
    return True
