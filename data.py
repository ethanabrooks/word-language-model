from io import open
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset


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
    def __init__(self, tokens: torch.Tensor, bptt: int):
        self.bptt = bptt
        self.tokens = tokens

    def __getitem__(self, index):
        seq_len = min(self.bptt, len(self.tokens) - 1 - index)
        return (
            self.tokens[index : index + seq_len],
            self.tokens[index + 1 : index + seq_len + 1],
        )

    def __len__(self):
        return len(self.tokens)


class DebugDataset(Dataset):
    def __init__(self, data: torch.Tensor, target: torch.Tensor, bptt: int):
        self.bptt = bptt
        self.target = target
        self.data = data

    @staticmethod
    def generate_targets(data: np.ndarray, p):
        vocab = 1 + data.max(initial=-np.inf)
        mapping = np.random.choice(2, (vocab, vocab), p=[p, 1 - p]).astype(bool)
        np.fill_diagonal(mapping, 0)
        prev = [None for _ in range(vocab)]
        for word in data:
            yield prev[word] or vocab
            for w in np.arange(vocab)[mapping[word]]:
                prev[w] = word

    def generate(self, path: Path, seed: int, size: int, n_tokens: int, p: float):
        np.random.seed(seed)
        data = np.random.choice(n_tokens, size)
        target = list(self.generate_targets(data, p))
        np.savez(str(path), data=data, target=target)

    def __getitem__(self, index):
        data = self.data[index : index + self.bptt]
        target = self.target[index : index + self.bptt]
        return data, target

    def __len__(self):
        return len(self.target) - self.bptt
