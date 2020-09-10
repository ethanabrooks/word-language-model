import itertools
import argparse
from pathlib import Path

import numpy as np


def generate(data: np.ndarray, p):
    vocab = 1 + data.max(initial=-np.inf)
    mapping = np.random.choice(2, (vocab, vocab), p=[p, 1 - p]).astype(bool)
    np.fill_diagonal(mapping, 0)
    prev = [None for _ in range(vocab)]
    for word in data:
        yield prev[word] or vocab
        for w in np.arange(vocab)[mapping[word]]:
            prev[w] = word


def main(path: Path, seed: int, size: int, vocab: int, p: float):
    np.random.seed(seed)
    data = np.random.choice(vocab, size)
    target = list(generate(data, p))
    np.savez(str(path), data=data, target=target)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--seed", "-s", type=int, default=0)
    PARSER.add_argument("--size", "-n", type=int, required=True)
    PARSER.add_argument("--vocab", "-v", type=int, required=True)
    PARSER.add_argument("-p", type=float, default=0.8)
    PARSER.add_argument("--path", type=Path, required=True)
    main(**vars(PARSER.parse_args()))
