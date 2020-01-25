# import os
from pathlib import Path
import glob
import pandas as pd
import time

import numpy as np
import torch
from torch.utils import data
from transformer import Constants
from torch.utils.data.dataset import Subset
from torch._utils import _accumulate

import concurrent.futures

# Math Dataset constants (from paper)

# input chars are selected from basic ASCII chars
VOCAB_SZ = 95
# questions have less than 160 chars (!)
MAX_QUESTION_SZ = 162
# answers have less than 30 chars (!)
MAX_ANSWER_SZ = 32


def random_split_dataset(ds, split_rate):
    """uses Torch utils to split and randomize data into train/val datasets"""
    size = len(ds)
    train_split = int(size * split_rate)
    val_split = size - train_split
    train_ds, val_ds = data.random_split(ds, [train_split, val_split])
    return train_ds, val_ds


def deterministic_split_dataset(ds, split_rate):
    """ Split data consistently into train/val datasets"""
    size = len(ds)
    train_split = int(size * split_rate)
    val_split = size - train_split

    lengths = [train_split, val_split]
    indices = sum(lengths).tolist()

    return [
        Subset(ds, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def np_encode_string(s, char0=ord(" ")):
    """converts a string into a numpy array of bytes
    (char0 - 1) is subtracted from all bytes values (0 is used for PAD)
    string is pre-pended with BOS and post-pended with EOS"""
    chars = np.array(list(s), dtype="S1").view(np.uint8)
    # normalize to 1 - 96, 0 being PAD
    chars = chars - char0 + 1

    chars = np.insert(chars, 0, Constants.BOS)
    chars = np.insert(chars, len(chars), Constants.EOS)
    return chars


def np_decode_string(chars, char0=ord(" ")):
    """converts a numpy array of bytes into a UTF-8 string
    (char0 - 1) is added to all bytes values (0 is used for PAD)
    BOS/EOS are removed before utf-8 decoding"""
    chars = chars.astype(np.uint8)
    chars = chars + char0 - 1
    chars = chars[:-1]
    chars = chars.tobytes()
    s = chars.decode("UTF-8")
    return s


class LazyFileMathDataset(data.Dataset):
    """Stream loads math dataset file in a lazy way (optional)
    pandas is used for naive streaming as Python doesn't provide any better tool for that critical feature"""

    def __init__(self, file, lazy_load=False, max_elements=None, log=False):
        self.file = Path(file)
        self.lazy_load = lazy_load
        self.max_elements = max_elements

        fn = self.file.name.replace(".txt", "")
        self.category, self.module = fn.split("__")

        if not self.lazy_load:
            self.build_dataset()
            if log:
                print(
                    f"Initialized MathDataset with file {self.file} (category:{self.category}, module:{self.module}) containing {self.qas.shape[0]} pairs of questions/answers"
                )
        else:
            self.qas = None
            if log:
                print(
                    f"Initialized MathDataset with file {self.file} (category:{self.category}, module:{self.module}) in lazy mode"
                )

    def _read_build_dataset(self):
        self.df = pd.read_csv(
            self.file, header=None, sep="\n", names=["qa"], engine="c"
        )
        self._build_dataset()

    def _build_dataset(self):
        if self.max_elements is not None:
            self.df_max = self.df.iloc[0 : self.max_elements * 2]
        else:
            self.df_max = self.df
        self.questions = self.df_max[0::2]
        self.questions.reset_index(inplace=True, drop=True)
        self.questions.rename(columns={"qa": "questions"}, inplace=True)
        self.answers = self.df_max[1::2]
        self.answers.reset_index(inplace=True, drop=True)
        self.answers.rename(columns={"qa": "answers"}, inplace=True)

        # Something like
        # Instead of a single dataset, you have an array of pandas datasets, one from each file .
        # So you just append the contents of each question and answer array.
        # The final 'qas' is the same.
        # OR alternatively, combining the output of pd.concats.
        # not sure why this isn't working to begin with
        # I guess it's using torch.data.ConcatDataset instead of something pandas?
        # I think what you'll need to do is just iterate over the LFMDs and get the ds.qas from each
        # so like
        # full_df = pd.DataFrame()
        # for category, modules in self.dfs.items():
        #   for module in modules:
        #       for typ, ds in module.items():
        #           if ["train-easy", "train-medium","train-hard"].contains(typ)
        #               full_df.append(ds.qas)
        # return

        self.qas = pd.concat([self.questions, self.answers], axis=1)

    def set_max_elements(self, max_elements):
        self.max_elements = max_elements
        if self.qas is None:
            self._read_build_dataset()
        else:
            self._build_dataset()

    def __getitem__(self, idx):
        if self.qas is None:
            raise ValueError("qas is none in __getitem__")
            # self._read_build_dataset()
        question, answer = self.qas.iloc[idx]
        return {
            "q": question,
            "q_enc": np_encode_string(question),
            "a": answer,
            "a_enc": np_encode_string(answer),
        }

    def __len__(self):
        if self.qas is None:
            raise ValueError("qas is none in __len__")
            # self._read_build_dataset()
        return self.qas.shape[0]


class MathDatasetManager:
    """A Math Dataset manager starting at root directory (like v1.0) to extract files and build torch datasets
    in a lazy loading and streamed way based on specific types/categories/modules presented in paper.

    It indexes difficulty/use-case types:
        - train-easy
        - train-medium
        - train-hard
        - interpolate
        - extrapolate

    and all categories:
        - algebra
        - numbers
        - polynomials
        - arithmetic
        - measurement
        - comparison
        - probability
        - calculus

    and all modules in those categories:
        - mul
        - add_or_sub_in_base
        - simplify_surd
        - mul_div_multiple
        - mixed
        - nearest_integer_root
        - div
        - add_or_sub
        - add_sub_multiple
        - add_sub_multiple_longer
        - mul_div_multiple_longer
        - div_big
        - mul_big
        - mixed_longer
        - add_or_sub_big
        - etc...
    """

    def __init__(self, root_dir, log=False):
        self.root_dir = Path(root_dir)

        self.dirs = {
            "train-easy": self.root_dir / "train-easy",
            "train-medium": self.root_dir / "train-medium",
            "train-hard": self.root_dir / "train-hard",
            "interpolate": self.root_dir / "interpolate",
            "extrapolate": self.root_dir / "extrapolate",
        }

        self.dfs = {}

        for k, dir in self.dirs.items():
            files = [ff for ff in glob.glob(str(dir) + "/**/*.txt", recursive=True)]
            for f in files:
                ds = LazyFileMathDataset(f, lazy_load=True, log=log)
                if ds.category not in self.dfs:
                    self.dfs[ds.category] = {}
                if ds.module not in self.dfs[ds.category]:
                    self.dfs[ds.category][ds.module] = {
                        "train-easy": {},
                        "train-medium": {},
                        "train-hard": {},
                        "interpolate": {},
                        "extrapolate": {},
                    }

                self.dfs[ds.category][ds.module][k] = ds

        print(
            f"initialized MultiFilesMathDataset with categories {list(self.dfs.keys())} and types {list(self.dirs.keys())}"
        )

    def get_types(self):
        """retrieves all math typesfor this multi-file dataset"""
        return self.dirs.keys()

    def get_categories(self):
        """retrieves all math problem categories in this multi-file dataset"""
        return self.dfs.keys()

    def get_modules_for_category(self, c):
        """retrieves all mathematical modules in a math problem category"""
        return self.dfs[c].keys()

    def _build_datasets_from_category(self, category, typ, max_elements=None):
        ds = []
        for k, m in self.dfs[category].items():
            if typ in m and hasattr(m[typ], "set_max_elements"):
                print(f"attempting to add module {category}/{k}/{typ}")
                m[typ].set_max_elements(max_elements)
                ds.append(m[typ])
                print(f"added module {category}/{k}/{typ}")
        return ds

    def build_dataset_from_category(self, category, typ, max_elements=None):
        """Build a dataset for all modules in a category"""
        print(f"adding category {category}/../{typ}")
        ds = self._build_datasets_from_category(
            category, typ, max_elements=max_elements
        )
        return data.ConcatDataset(ds)

    def build_dataset_from_categories(self, categories, typ, max_elements=None):
        """Build a dataset for all modules in several categories"""
        ds = []
        for c in categories:
            print(f"adding category.. {c}/../{typ}")
            dss = self._build_datasets_from_category(c, typ, max_elements=max_elements)
            ds.extend(dss)

        return data.ConcatDataset(ds)

    def build_dataset_full(self):
        """Builds the entire dataset"""
        ds = []
        for typ in ["train-easy", "train-medium", "train-hard"]:
            for c in [
                "algebra",
                "numbers",
                "polynomials",
                "arithmetic",
                "measurement",
                "comparison",
                "probability",
                "calculus",
            ]:
                print(f"adding category {c}/../{typ}")
                dss = self._build_datasets_from_category(c, typ)
                ds.extend(dss)
        return data.ConcatDataset(ds)

    def build_dataset_from_level(self, level):
        """Builds the dataset for a level"""
        ds = []
        for c in [
            "algebra",
            "numbers",
            "polynomials",
            "arithmetic",
            "measurement",
            "comparison",
            "probability",
            "calculus",
        ]:
            print(f"adding category {c}/../{level}")
            dss = self._build_datasets_from_category(c, level)
            ds.extend(dss)
        return data.ConcatDataset(ds)

    def build_dataset_from_module(self, category, module, typ, max_elements=None):
        """Build a dataset from a single module in a category"""
        self.dfs[category][module][typ].set_max_elements(max_elements)
        return self.dfs[category][module][typ]

    def build_dataset_from_modules(self, category, modules, typ, max_elements=None):
        """Build a dataset from several modules in a category"""
        ds = []
        for module in modules:
            self.dfs[category][module][typ].set_max_elements(max_elements)
            ds.append(self.dfs[category][module][typ])
        return data.ConcatDataset(ds)


class FullDatasetManager(data.Dataset):
    def __init__(self, root_dir, max_elements=None, deterministic=False, start_epoch=0):
        self.root_dir = Path(root_dir)
        self.full_df = None
        self.max_elements = max_elements

        self.dirs = {
            "train-easy": self.root_dir / "train-easy",
            "train-medium": self.root_dir / "train-medium",
            "train-hard": self.root_dir / "train-hard",
        }

        print(f"Loading training data with max_elements: {self.max_elements}")
        start = time.time()
        data = {"questions": [], "answers": [], "original_index": []}
        # all_questions = []
        # all_answers = []
        files = [
            ff
            for key, dir in self.dirs.items()
            for ff in glob.glob(str(dir) + "/**/*.txt", recursive=True)
        ]
        print(f"File count: {len(files)}")
        data_index = 0
        if deterministic:
            for questions, answers in map(self._getQuestionsAnswersFromFile, files):
                data["questions"].extend(questions)
                data["answers"].extend(answers)
                data["original_index"] = data_index
                data_index += 1
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for questions, answers in executor.map(
                    self._getQuestionsAnswersFromFile, files
                ):
                    data["questions"].extend(questions)
                    data["answers"].extend(answers)
                    data["original_index"] = data_index
                    data_index += 1

        self.full_df = pd.DataFrame(data)
        for i in range(start_epoch + 1):
            self.shuffleData()

        print(
            f"Took {time.time() - start} seconds to initialize full dataset of length {self.full_df.shape[0]}. Deterministic: {deterministic}"
        )

    def shuffleData(self):
        # Will shuffle deterministically if numpy seed is set
        start = time.time()
        self.full_df = self.full_df.reindex(np.random.permutation(self.full_df.index))
        print(f"Speed of shuffling dataset: {(time.time() - start) * 1000}")

    def _getQuestionsAnswersFromFile(self, filepath):
        count = 0
        with open(filepath) as datafile:
            questions = []
            answers = []
            for line in datafile:
                line = line.rstrip("\n")
                if self.max_elements is not None and count == (2 * self.max_elements):
                    return questions, answers
                if count % 2 == 0:
                    questions.append(line)
                else:
                    answers.append(line)
                count += 1
            print(f"{len(questions)} questions in {filepath}")
            return questions, answers

    def __getitem__(self, idx):
        if self.full_df is None:
            raise ValueError("full_df is none in __getitem__")
        question, answer, _ = self.full_df.iloc[idx]
        return {
            "q": question,
            "q_enc": np_encode_string(question),
            "a": answer,
            "a_enc": np_encode_string(answer),
        }

    def __len__(self):
        if self.full_df is None:
            raise ValueError("full_df is none in __len__")
        return self.full_df.shape[0]


# Core collate function
def question_answer_to_position_batch_collate_fn(qas):
    """ Gather + Pad the question/answer to the max seq length in batch """

    max_q_len = max(len(qa["q_enc"]) for qa in qas)
    max_a_len = max(len(qa["a_enc"]) for qa in qas)

    batch_qs = []
    batch_as = []
    for qa in qas:
        batch_qs.append(
            np.pad(
                qa["q_enc"],
                (0, max_q_len - len(qa["q_enc"])),
                mode="constant",
                constant_values=Constants.PAD,
            )
        )
        batch_as.append(
            np.pad(
                qa["a_enc"],
                (0, max_a_len - len(qa["a_enc"])),
                mode="constant",
                constant_values=Constants.PAD,
            )
        )

    batch_qs_pos = np.array(
        [
            [pos_i + 1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(q)]
            for q in batch_qs
        ]
    )

    batch_as_pos = np.array(
        [
            [pos_i + 1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(a)]
            for a in batch_as
        ]
    )

    batch_qs = torch.LongTensor(batch_qs)
    batch_qs_pos = torch.LongTensor(batch_qs_pos)

    batch_as = torch.LongTensor(batch_as)
    batch_as_pos = torch.LongTensor(batch_as_pos)

    return batch_qs, batch_qs_pos, batch_as, batch_as_pos


# def question_answer_to_batch_collate_fn(qas):
#     """ Gather + Pad the question/answer to the max seq length in batch """

#     max_q_len = max(len(qa["q_enc"]) for qa in qas)
#     max_a_len = max(len(qa["a_enc"]) for qa in qas)

#     batch_qs = []
#     batch_as = []
#     # batch_pos = []
#     for qa in qas:
#         batch_qs.append(
#             np.pad(
#                 qa["q_enc"],
#                 (0, max_q_len - len(qa["q_enc"])),
#                 mode="constant",
#                 constant_values=Constants.PAD,
#             )
#         )
#         batch_as.append(
#             np.pad(
#                 qa["a_enc"],
#                 (0, max_a_len - len(qa["a_enc"])),
#                 mode="constant",
#                 constant_values=Constants.PAD,
#             )
#         )

#     batch_qs = torch.LongTensor(batch_qs)
#     batch_as = torch.LongTensor(batch_as)

#     return batch_qs, batch_as


def question_to_position_batch_collate_fn(qs):
    """ Gather + Pad the question to the max seq length in batch """

    max_q_len = max(len(q) for q in qs)

    batch_qs = []
    for q in qs:
        batch_qs.append(
            np.pad(
                q,
                (0, max_q_len - len(q)),
                mode="constant",
                constant_values=Constants.PAD,
            )
        )

    batch_qs_pos = np.array(
        [
            [pos_i + 1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(q)]
            for q in batch_qs
        ]
    )

    batch_qs = torch.LongTensor(batch_qs)
    batch_qs_pos = torch.LongTensor(batch_qs_pos)

    return batch_qs, batch_qs_pos
