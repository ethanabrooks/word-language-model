import math
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader

import models
import ours
from data import Corpus, LMDataset, DebugDataset


class Aggregator(ABC):
    def __init__(self):
        self.values = defaultdict(list)

    def update(self, **values):
        for k, v in values.items():
            self.values[k].append(v)


class MeanAggregator(Aggregator):
    def items(self):
        for k, v in self.values.items():
            yield k, torch.mean(torch.stack(v)).item()


def run(
    batch_size: int,
    bptt: int,
    clip: float,
    cuda: bool,
    data: Path,
    dry_run: bool,
    em_size: int,
    epochs: int,
    log_interval: int,
    lr: float,
    model: str,
    n_heads: int,
    report: callable,
    save: Path,
    seed: int,
    tied: bool,
    load: Optional[Path] = None,
    onnx_export: Optional[Path] = None,
    **kwargs,
):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    cuda = cuda and torch.cuda.is_available()

    device = torch.device("cuda" if cuda else "cpu")
    print("Running with device:", device)

    ###############################################################################
    # Load data
    ###############################################################################

    eval_batch_size = 10
    if data.name == "debug.npz":
        if not data.exists():
            DebugDataset.generate(
                data, seed=seed, n_seq=10000, seq_len=bptt, n_tokens=10, p=0.8
            )
        dataset = DebugDataset(data, device)
        assert bptt == dataset.bptt
        ntokens = dataset.n_tokens + 1
        n_seq = len(dataset)
        size_valid = int(n_seq * 0.2)
        size_test = int(n_seq * 0.1)
        train_data, val_data, test_data = torch.utils.data.random_split(
            dataset, [n_seq - size_test - size_valid, size_valid, size_test]
        )
    else:
        corpus = Corpus(data)
        train_data = LMDataset(
            corpus.train, bptt, batch_size=batch_size, device=device
        )  # [104431, 20]
        val_data = LMDataset(
            corpus.valid, bptt, batch_size=batch_size, device=device
        )  # [21764, 10]
        test_data = LMDataset(
            corpus.test, bptt, batch_size=batch_size, device=device
        )  # [24556, 10]
        ntokens = len(corpus.dictionary)

    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    ###############################################################################
    # Build the model
    ###############################################################################
    em_size = (em_size // n_heads) * n_heads
    kwargs.update(n_tokens=ntokens, em_size=em_size)
    recurrent = False
    if model == "transformer":
        model = models.TransformerModel(n_head=n_heads, **kwargs).to(device)
    elif model == "ours":
        model = ours.TransformerModel(n_head=n_heads, **kwargs).to(device)
    else:
        model = models.RNNModel(model, tied, **kwargs).to(device)
        recurrent = True
    if load is not None:
        with load.open("rb") as f:
            model.load_state_dict(torch.load(f))
            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            # Currently, only rnn model supports flatten_parameters function.
            if recurrent:
                model.rnn.flatten_parameters()

    ###############################################################################
    # Training code
    ###############################################################################

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        hidden = model.init_hidden(eval_batch_size) if recurrent else None
        with torch.no_grad():
            for (inputs, targets) in data_source:
                targets = targets.flatten()
                if hidden is None:
                    output = model(inputs)
                    output = output.reshape(-1, ntokens)
                else:
                    output, hidden = model(inputs, hidden)
                    hidden = repackage_hidden(hidden)
                yield len(inputs) * criterion(output, targets).item()

    criterion = nn.NLLLoss()

    def train():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        hidden = model.init_hidden(eval_batch_size) if recurrent else None
        for i, (inputs, targets) in enumerate(train_data):
            targets = targets.flatten()
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            if hidden is None:
                outputs = model(inputs)
                outputs = outputs.reshape(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)
                outputs, hidden = model(inputs, hidden)
            is_accurate = outputs.max(-1).indices == targets
            assert isinstance(is_accurate, torch.Tensor)
            accuracy = torch.mean(is_accurate.float())
            total_acc += accuracy
            loss = criterion(outputs, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

            total_loss += loss.item()
            logs = dict(epoch=epoch, batches=i)
            means = dict(accuracy=accuracy, loss=loss)
            writes = dict(inputs=inputs, outputs=outputs, targets=targets)
            yield logs, means, writes
            if dry_run:
                break

    def export_onnx(path, bsz, seq_len):
        print(
            "The model is also exported in ONNX format at {}".format(
                onnx_export.absolute()
            )
        )
        model.eval()
        dummy_input = torch.LongTensor(seq_len * bsz).zero_().view(-1, bsz).to(device)
        hidden = model.init_hidden(bsz)
        torch.onnx.export(model, (dummy_input, hidden), str(path))

    # Loop over epochs.
    best_val_loss = None
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, epochs + 1):
            aggregator = MeanAggregator()
            for batch, (to_log, to_mean, to_write) in enumerate(train()):
                aggregator.update(**to_mean)
                if batch % log_interval == 0 and batch > 0:
                    report(**to_log, **dict(aggregator.items()))
                    aggregator = MeanAggregator()

            val_loss = np.mean(list(evaluate(val_data)))
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with save.open("wb") as f:
                    torch.save(model.state_dict(), f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
    # Load the best saved model.
    with save.open("rb") as f:
        model.load_state_dict(torch.load(f))
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if recurrent:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = np.mean(list(evaluate(test_data)))
    report(test_loss=test_loss, test_ppl=math.exp(test_loss))
    if onnx_export:
        # Export the model in ONNX format.
        export_onnx(onnx_export, bsz=1, seq_len=bptt)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
