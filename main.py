# coding: utf-8
import argparse
import numpy as np
import math
from pathlib import Path
from typing import Optional

import ray
import torch
import torch.nn as nn
import torch.onnx
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from data import Corpus

import models
import ours


def add_arguments(parser):
    parser.add_argument(
        "--batch_size", type=int, default=20, metavar="N", help="batch size"
    )
    parser.add_argument("--bptt", type=int, default=35, help="sequence length")
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument(
        "--config",
        type=str,
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
        default=0.2,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="verify the code and the model"
    )
    parser.add_argument(
        "--em-size", type=int, default=200, help="size of word embeddings"
    )
    parser.add_argument("--epochs", type=int, default=40, help="upper epoch limit")
    parser.add_argument(
        "--gpus-per-trial",
        "-g",
        type=int,
        default=1,
        help="GPU resources to allocate per trial. Note that GPUs will not be assigned unless you specify them.",
    )
    parser.add_argument(
        "--log-interval", type=int, default=200, metavar="N", help="report interval"
    )
    parser.add_argument("--lr", type=float, default=20, help="initial learning rate")
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
        default=2,
        help="the number of heads in the encoder/decoder of the transformer model",
    )
    parser.add_argument(
        "--n-hid", type=int, default=200, help="number of hidden units per layer"
    )
    parser.add_argument("--n-layers", type=int, default=2, help="number of layers")
    parser.add_argument(
        "--n-samples",
        type=int,
        help="Number of times to sample from the hyperparameter space. If not set, tune will be run in local mode.",
    )
    parser.add_argument(
        "--save", type=str, default="model.pt", help="path to save the final model"
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument(
        "--tied", action="store_true", help="tie the word embedding and softmax weights"
    )


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def run(
    batch_size: int,
    bptt: int,
    clip: float,
    cuda: bool,
    data: Path,
    dropout: float,
    dry_run: bool,
    em_size: int,
    epochs: int,
    log_interval: int,
    lr: float,
    model: str,
    n_head: int,
    n_hid: int,
    n_layers: int,
    save: str,
    seed: int,
    tied: bool,
    onnx_export: Optional[Path] = None,
):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda"
            )

    device = torch.device("cuda" if cuda else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    def batchify(data_source, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        n_batch = data_source.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data_source = data_source.narrow(0, 0, n_batch * bsz)
        # Evenly divide the data across the bsz batches.
        data_source = data_source.view(bsz, -1).t().contiguous()
        return data_source.to(device)

    debug_dataset = "debug" in str(data)

    eval_batch_size = 10
    if debug_dataset:

        def load(fname, bsz):
            arrays = np.load(str(Path(data, fname))).values()
            return [batchify(torch.tensor(x, device=device), bsz) for x in arrays]

        train_data = load("train.npz", batch_size)
        val_data = load("valid.npz", eval_batch_size)
        test_data = load("test.npz", eval_batch_size)
        n_tokens = 1 + int(
            max((max(d.max(), t.max()) for d, t in [train_data, val_data, test_data]))
        )
    else:
        corpus = Corpus(data)
        train_data = batchify(corpus.train, batch_size)  # [104431, 20]
        val_data = batchify(corpus.valid, eval_batch_size)  # [21764, 10]
        test_data = batchify(corpus.test, eval_batch_size)  # [24556, 10]

        ###############################################################################
        # Build the model
        ###############################################################################

        n_tokens = len(corpus.dictionary)

    def size_data(data):
        if debug_dataset:
            return data[0].size(0)
        else:
            return data.size(0)

    recurrent = model not in ["transformer", "ours"]
    if model == "transformer":
        model = models.TransformerModel(
            n_tokens, em_size, n_head, n_hid, n_layers, dropout
        ).to(device)
    elif model == "ours":
        model = ours.TransformerModel(
            n_tokens, em_size, n_head, n_hid, n_layers, dropout
        ).to(device)
    else:
        model = models.RNNModel(
            model,
            n_tokens,
            em_size,
            n_hid,
            n_layers,
            dropout,
            tied,
        ).to(device)

    ###############################################################################
    # Training code
    ###############################################################################

    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(source, i):
        if debug_dataset:
            data, target = source
            seq_len = min(bptt, len(data) - 1 - i)
            return data[i : i + seq_len], target[i : i + seq_len].flatten()
        else:
            seq_len = min(bptt, len(source) - 1 - i)
            data = source[i : i + seq_len]
            target = source[i + 1 : i + 1 + seq_len].view(-1)
            return data, target

    criterion = nn.NLLLoss()

    # Loop over epochs.
    best_val_loss = None

    def train():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.0
        hidden = model.init_hidden(batch_size) if recurrent else None
        for batch, i in enumerate(range(0, size_data(train_data) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            if not recurrent:
                output = model(data)
                output = output.view(-1, n_tokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            loss = criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

            total_loss += loss.item()

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                tune.report(
                    epoch=epoch, batch=batch, loss=cur_loss, ppl=math.exp(cur_loss)
                )
                total_loss = 0
            if dry_run:
                break

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.0
        if recurrent:
            hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, size_data(data_source) - 1, bptt):
                data, targets = get_batch(data_source, i)
                if not recurrent:
                    output = model(data)
                    output = output.view(-1, n_tokens)
                else:
                    output, hidden = model(data, hidden)
                    hidden = repackage_hidden(hidden)
                total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)

    def export_onnx(path, batch_size, seq_len):
        print(
            "The model is also exported in ONNX format at {}".format(
                onnx_export.absolute()
            )
        )
        model.eval()
        dummy_input = (
            torch.LongTensor(seq_len * batch_size)
            .zero_()
            .view(-1, batch_size)
            .to(device)
        )
        hidden = model.init_hidden(batch_size)
        torch.onnx.export(model, (dummy_input, hidden), path)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, epochs + 1):
            # epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            tune.report(val_loss=val_loss)
            if not best_val_loss or val_loss < best_val_loss:
                with open(save, "wb") as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    # Load the best saved model.
    with open(save, "rb") as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if recurrent:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data)
    tune.report(test_loss=test_loss, test_ppl=math.exp(test_loss))

    if onnx_export:
        # Export the model in ONNX format.
        export_onnx(onnx_export, batch_size=1, seq_len=bptt)


def main(
    config: Optional[dict],
    cpus_per_trial: int,
    data: Path,
    epochs: int,
    gpus_per_trial: int,
    n_samples: int,
    name: str,
    **kwargs,
):
    if config is None:
        config = dict()

    for k, v in kwargs.items():
        if v is not None:
            config[k] = v

    config.update(epochs=epochs, data=data.absolute())
    local_mode = n_samples is None
    ray.init(dashboard_host="127.0.0.1", local_mode=local_mode)
    if local_mode:
        kwargs = dict()
    else:
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
        stop=dict(training_iteration=epochs),
        **kwargs,
    )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model"
    )
    add_arguments(PARSER)
    main(**vars(PARSER.parse_args()))
