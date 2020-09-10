import json
from pathlib import Path

from hyperopt import hp


def get_config(name):
    path = Path("configs", name).with_suffix(".json")
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return configs[name]


default = dict(
    batch_size=20,
    bptt=35,
    clip=0.25,
    dropout=0.2,
    em_size=200,
    forward_scan=False,
    last_col_1=False,
    lr=20,
    n_head=2,
    n_hid=200,
    n_layers=2,
)

search = dict(
    batch_size=hp.choice("batch_size", [5, 10, 20, 30]),
    bptt=hp.choice("bptt", [25, 35, 45]),
    clip=hp.choice("clip", [0.2, 0.25, 0.3]),
    dropout=hp.choice("dropout", [0.05, 0.1, 0.2]),
    em_size=hp.choice("em_size", [50, 100, 200, 250, 300]),
    forward_scan=hp.choice("forward_scan", [True, False]),
    last_col_1=hp.choice("last_col_1", [True, False]),
    lr=hp.choice("lr", [5, 10, 20, 30]),
    n_head=hp.choice("n_head", [1, 2, 3, 4, 5, 6]),
    n_hid=hp.choice("n_hid", [100, 200, 250, 300]),
    n_layers=hp.choice("n_layers", [1, 2, 3, 4]),
)

configs = dict(search=search, default=default)
