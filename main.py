# coding: utf-8
import argparse
from pathlib import Path
from typing import Optional, List
from pprint import pprint

import ray
from hyperopt.pyll import Apply
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from config import get_config, default

from train import run


def add_arguments(parser):
    parser.add_argument("--batch-size", type=int, metavar="N", help="batch size")
    parser.add_argument("--bptt", type=int, help="sequence length")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="use CUDA")
    parser.add_argument("--clip", type=float, help="gradient clipping")
    parser.add_argument(
        "--config",
        type=get_config,
        default=default,
        help="config file to use for Experiment",
    )
    parser.add_argument(
        "--cpus-per-trial",
        "-c",
        type=int,
        default=6,
        help="CPU resources to allocate per trial.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default="./data/wikitext-2",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify the code and the model",
    )
    parser.add_argument("--em-size", type=int, help="size of word embeddings")
    parser.add_argument("--epochs", type=int, default=40, help="upper epoch limit")
    parser.add_argument(
        "--gpus-per-trial",
        "-g",
        type=int,
        default=1,
        help="GPU resources to allocate per trial. Note that GPUs will not be assigned unless you specify them.",
    )
    parser.add_argument("--local-mode", action="store_true")
    parser.add_argument(
        "--log-interval", type=int, default=200, metavar="N", help="report interval"
    )
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument(
        "--onnx-export",
        type=Path,
        help="path to export the final model in onnx format",
    )
    parser.add_argument(
        "--model",
        choices="RNN_TANH RNN_RELU LSTM GRU transformer ours".split(),
        default="LSTM",
        help="type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, transformer, ours)",
    )
    parser.add_argument(
        "--name",
        help="Name of experiment",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        help="the number of heads in the encoder/decoder of the transformer model",
    )
    parser.add_argument("--n-hid", type=int, help="number of hidden units per layer")
    parser.add_argument("--n-layers", type=int, help="number of layers")
    parser.add_argument(
        "--n-samples",
        type=int,
        help="Number of times to sample from the hyperparameter space. If not set, tune will be run in local mode.",
    )
    parser.add_argument(
        "--save", type=Path, default="model.pt", help="path to save the final model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        dest="seeds",
        default=[],
        nargs="*",
        help="random seed",
    )
    parser.add_argument(
        "--tied", action="store_true", help="tie the word embedding and softmax weights"
    )


def main(
    config: Optional[dict],
    cpus_per_trial: int,
    data: Path,
    gpus_per_trial: int,
    local_mode: bool,
    n_samples: int,
    name: str,
    seeds: List[int],
    **kwargs,
):
    for k, v in kwargs.items():
        if v is not None:
            config[k] = v

    config.update(data=data.absolute())
    if len(seeds) == 0:
        seed = 0
    elif len(seeds) == 1:
        seed = seeds[0]
    else:
        seed = tune.grid_search(seeds)
    config.update(seed=seed)
    if n_samples or local_mode:
        config.update(report=tune.report)
        ray.init(dashboard_host="127.0.0.1", local_mode=local_mode)
        kwargs = dict()
        if any(isinstance(v, Apply) for v in config.values()):
            kwargs = dict(
                search_alg=HyperOptSearch(config, metric="test_loss"),
                num_samples=n_samples,
            )

        def _run(c):
            run(**c)

        tune.run(
            _run,
            name=name,
            config=config,
            resources_per_trial=dict(gpu=gpus_per_trial, cpu=cpus_per_trial),
            **kwargs,
        )
    else:

        def report(**kwargs):
            pprint(kwargs)

        config.update(report=report)
        run(**config)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model"
    )
    add_arguments(PARSER)
    main(**vars(PARSER.parse_args()))
