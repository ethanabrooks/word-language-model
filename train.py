import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ray import tune
from torch import nn as nn

import models
import ours
from data import Corpus


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
        data_source = data_source.view(bsz, -1).t().contiguous().to(device)
        targets = data_source.roll(1, 0)
        return data_source, targets

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

        n_tokens = 1 + len(corpus.dictionary)

    def size_data(data):
        if debug_dataset:
            return data[0].size(0)
        else:
            return data[0].size(0)

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
            model, n_tokens, em_size, n_hid, n_layers, dropout, tied,
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

    def get_batch(data, target, i):
        if debug_dataset:
            data, target = source
            seq_len = min(bptt, len(data) - 1 - i)
            return data[i : i + seq_len], target[i : i + seq_len].flatten()
        else:
            seq_len = min(bptt, len(data) - 1 - i)
            data = data[i : i + seq_len]
            target = target[i : i + seq_len]
            target[0] = n_tokens - 1
            return data, target.view(-1)

    criterion = nn.NLLLoss()

    # Loop over epochs.
    best_val_loss = None

    def train():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        hidden = model.init_hidden(batch_size) if recurrent else None
        for batch, i in enumerate(range(0, size_data(train_data) - 1, bptt)):
            data, targets = get_batch(*train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            if not recurrent:
                output = model(data)
                output = output.view(-1, n_tokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            total_accuracy += torch.mean((output.max(-1).indices == targets).float())
            loss = criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

            total_loss += loss.item()

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                cur_accuracy = total_accuracy / log_interval
                tune.report(
                    epoch=epoch,
                    batch=batch,
                    loss=cur_loss,
                    ppl=math.exp(cur_loss),
                    accuracy=float(cur_accuracy),
                )
                total_accuracy = 0
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


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
