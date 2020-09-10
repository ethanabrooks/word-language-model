import math
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ray import tune
from torch import nn as nn
from torch.utils.data import DataLoader

import models
import ours
from data import Corpus, LMDataset


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

    def get_batches(data_source):
        for batch, i in enumerate(range(0, size_data(data_source) - 1, bptt)):
            yield get_batch(train_data, i)

    criterion = nn.NLLLoss()

    # Loop over epochs.
    best_val_loss = None

    def train():
        # Turn on training mode which enables dropout.
        model.train()
        hidden = model.init_hidden(batch_size) if recurrent else None
        for batch, (data, targets) in enumerate(get_batches(train_data)):
            data = data.to(device)
            targets = targets.to(device)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            if not recurrent:
                output = model(data)
                output = output.view(-1, n_tokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            is_accurate = output.max(-1).indices == targets
            assert isinstance(is_accurate, torch.Tensor)
            accuracy = torch.mean(is_accurate.float())
            loss = criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

            # TODO: only save a subset of data/output/targets
            info = dict(epoch=epoch, batch=batch)
            mean_info = dict(loss=loss.item(), accuracy=accuracy.item())
            write_info = dict(
                data=data[:, 0],
                output=output.view(*data.shape, -1)[:, 0],
                targets=targets.view(data.shape)[:, 0],
            )
            yield info, mean_info, write_info
            if dry_run:
                break

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        if recurrent:
            hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for (data, targets) in get_batches(data_source):
                if not recurrent:
                    output = model(data)
                    output = output.view(-1, n_tokens)
                else:
                    output, hidden = model(data, hidden)
                    hidden = repackage_hidden(hidden)
                yield len(data) * criterion(output, targets).item()

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
            means = MeanAggregator()
            for i, (to_log, to_mean, to_write) in enumerate(train()):
                means.update(**to_mean)
                if (i + 1) % log_interval == 0:
                    tune.report(**to_log)
                    tune.report(**dict(means.items()))
                    with tune.checkpoint_dir(epoch) as path:
                        np.savez(path, to_write)
                    means = MeanAggregator()

            val_loss = np.mean(list(evaluate(val_data)))
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
    test_loss = np.mean(list(evaluate(test_data)))
    tune.report(test_loss=test_loss, test_ppl=math.exp(test_loss))

    if onnx_export:
        # Export the model in ONNX format.
        export_onnx(onnx_export, batch_size=1, seq_len=bptt)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
