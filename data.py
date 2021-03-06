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
    def __init__(self, path: Path, device):
        arrays = np.load(str(path))
        self.target = torch.Tensor(arrays["target"]).long().to(device)
        self.data = torch.Tensor(arrays["data"]).long().to(device)
        self.mapping = torch.Tensor(arrays["mapping"]).to(device)
        self.n_tokens = 1 + self.mapping.size(0)
        # print("MAPPING")
        # for i, row in enumerate(arrays["mapping"].T):
        # print(i, np.arange(self.n_tokens - 1)[row])
        self.bptt = self.data.size(1)

    def print_mapping(self):
        for i, row in enumerate(self.mapping.T):
            print(i, np.arange(self.n_tokens - 1)[row])

    @staticmethod
    def generate_targets(data: np.ndarray, mapping: np.ndarray, n_tokens: int):
        np.fill_diagonal(mapping, 0)
        for sentence in tqdm(data):
            prev = [None for _ in range(n_tokens)]
            for word in sentence:
                target = prev[word]
                if target is None:
                    target = n_tokens
                yield target
                for w in np.arange(n_tokens)[mapping[word]]:
                    prev[w] = word

    @classmethod
    def generate(
        cls, path: Path, seed: int, n_seq: int, seq_len: int, n_tokens: int, p: float
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed)
        data = np.random.choice(n_tokens, (n_seq, seq_len))
        mapping = np.random.choice(2, (n_tokens, n_tokens), p=[p, 1 - p]).astype(bool)
        target = np.array(list(cls.generate_targets(data, mapping, n_tokens))).reshape(
            n_seq, seq_len
        )
        np.savez(str(path), data=data, target=target, mapping=mapping)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        return data, target

    def __len__(self):
        return len(self.data)
