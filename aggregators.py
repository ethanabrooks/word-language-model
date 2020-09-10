from abc import ABC
from collections import defaultdict

import numpy as np


class Aggregator(ABC):
    def __init__(self):
        self.values = defaultdict(list)

    def update(self, **values):
        for k, v in values.items():
            self.values[k].append(v)


class MeanAggregator(Aggregator):
    def items(self):
        for k, v in self.values.items():
            yield k, np.mean(v)
