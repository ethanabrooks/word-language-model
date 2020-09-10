from io import open
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path: Path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(Path(path, "train.txt"))
        self.valid = self.tokenize(Path(path, "valid.txt"))
        self.test = self.tokenize(Path(path, "test.txt"))

    def tokenize(self, path: Path) -> torch.Tensor:
        """Tokenizes a text file."""
        assert path.exists(), f"{path} does not exist"
        # Add words to the dictionary
        with open(str(path), "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ["<eos>"]
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


class LMDataset(Dataset):
    def __init__(self, data_source: torch.Tensor, bptt: int, batch_size, device):
        self.bptt = bptt
        # Work out how cleanly we can divide the dataset into bsz parts.
        n_batch = data_source.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data_source = data_source.narrow(0, 0, n_batch * batch_size)
        self.tokens = data_source.to(device)

    def __getitem__(self, index):
        i = index * self.bptt
        data = self.tokens[i : i + self.bptt]
        target = self.tokens[i + 1 : i + 1 + self.bptt].view(-1)
        return data, target

    def __len__(self):
        return len(self.tokens) // self.bptt


class DebugDataset(Dataset):
    def __init__(self, data: torch.Tensor, target: torch.Tensor):
        self.target = target
        self.data = data

    def __getitem__(self, index):
        seq_len = min(self.bptt, len(self.tokens) - 1 - index)
        return (
            self.tokens[index : index + seq_len],
            self.tokens[index + 1 : index + seq_len + 1],
        )

    def __len__(self):
        return len(self.tokens)
