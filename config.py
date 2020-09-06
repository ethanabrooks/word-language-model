import json
from pathlib import Path

from hyperopt import hp


def get_config(name):
    path = Path("configs", name).with_suffix(".json")
    if path.exists():
        with path.open() as f:
            config = json.load(f)
            del config["use_tune"]
            return config
    return configs[name]


search = dict(
    batch_size=hp.choice("batch_size", [10, 20, 30]),
    bptt=hp.choice("bptt", [25, 35, 45]),
    clip=hp.choice("clip", [0.2, 0.25, 0.3]),
    dropout=hp.choice("dropout", [0.1, 0.2]),
    n_head=hp.choice("n_head", [1, 2, 3]),
    em_size=hp.choice("n_hid", [100, 200]),
    n_hid=hp.choice("n_layers", [1, 2, 3]),
)

configs = dict(search=search)
