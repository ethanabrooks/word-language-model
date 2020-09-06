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
    cuda = cuda and torch.cuda.is_available()

    device = torch.device("cuda" if cuda else "cpu")
    print("Running with device:", device)

    ###############################################################################
    # Load data
    ###############################################################################

    debug_dataset = "debug" in str(data)

    eval_batch_size = 10
    if debug_dataset:

        def build_data_loader(fname):
            arrays = np.load(str(Path(data, fname)))
            tensors = {k: torch.Tensor(v).long().to(device) for k, v in arrays.items()}
            return DataLoader(DebugDataset(**tensors, bptt=bptt), batch_size=batch_size)

        train_data = build_data_loader("train.npz")
        val_data = build_data_loader("valid.npz")
        test_data = build_data_loader("test.npz")
        n_tokens = 11
    else:
        corpus = Corpus(data)
        train_data = DataLoader(LMDataset(corpus.train, bptt), batch_size=batch_size)
        val_data = DataLoader(LMDataset(corpus.valid, bptt), batch_size=batch_size)
        test_data = DataLoader(LMDataset(corpus.test, bptt), batch_size=batch_size)

        n_tokens = len(corpus.dictionary)

    ###############################################################################
    # Build the model
    ###############################################################################

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

    criterion = nn.NLLLoss()
    best_val_loss = None

    def train():
        # Turn on training mode which enables dropout.
        model.train()
        hidden = model.init_hidden(batch_size) if recurrent else None
        for batch, (data, targets) in enumerate(train_data):
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            targets = targets.view(-1)
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
            for i, (data, targets) in enumerate(data_source):
                targets = targets.view(-1)
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
        # Loop over epochs.
        for epoch in range(1, epochs + 1):
            # epoch_start_time = time.time()
            means = MeanAggregator()
            for i, (info, mean_info, write_info) in enumerate(train()):
                means.update(**mean_info)
                if (i + 1) % log_interval == 0:
                    tune.report(**info, **dict(means.items()))
                    with tune.checkpoint_dir(epoch) as path:
                        np.savez(
                            path,
                            **{
                                k: v.detach().cpu().numpy()
                                for k, v in write_info.items()
                            },
                        )

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
