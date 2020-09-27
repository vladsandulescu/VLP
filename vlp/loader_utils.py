from random import randint, shuffle
from random import random as rand
import pickle
import json
from collections import namedtuple
import torch
import torch.nn as nn
import unicodedata
from multiprocessing import Lock


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def batch_list_to_batch_tensors_paired(batch):
    batch_tensors = []
    paired_batch = [x for t in batch for x in t[:-1]]
    label = [v for t in batch for k, v in t[-1].items()]
    for x in zip(*paired_batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    batch_tensors.append(torch.tensor(label, dtype=torch.long))
    return batch_tensors


def batch_list_to_batch_tensors_paired_eval(batch):
    batch_tensors = []
    paired_batch = [x for t in batch for x in t]
    for x in zip(*paired_batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()
        self.mask_same_word = None
        self.skipgram_prb = None
        self.skipgram_size = None

    def __call__(self, instance):
        raise NotImplementedError
