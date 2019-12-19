from tqdm import tqdm
import torch

from .helpers import Param


def train_epoch(dl, model, opt, loss_fn, device, show_progress):
    total_loss = 0
    model.train()
    for example, label in tqdm(dl) if show_progress else dl:
        example, label = example.to(device), label.to(device)
        opt.zero_grad()
        output = model(example)
        loss = loss_fn(output, label)
        loss.backward()
        opt.step()
        total_loss += loss.detach().item()
    return (total_loss, total_loss / len(dl))


def pct_error(output, label):
    return torch.div(
        torch.div(
            torch.abs(torch.sub(output, label)),
            label
        ).sum(),
        label.shape[0]
    ).item()

def validate(dl, model, loss_fn, device):
    total_loss = 0
    total_pct_error = 0
    metrics = []
    model.eval()
    for example, label in dl:
        example, label = example.to(device), label.to(device)
        output = model(example)
        loss = loss_fn(output, label)
        total_loss += loss.item()
        total_pct_error += pct_error(output.detach().cpu(), label.detach().cpu())
    return (total_loss, total_loss / len(dl), total_pct_error / len(dl))


def train(dl_train, dl_val, opt_func, loss_fn, model, epochs, device, show_progress):
    model = model.to(device)
    opt = opt_func(model.parameters())
    metrics = []
    for epoch in tqdm(range(epochs)) if show_progress else range(epochs):
        train_loss, train_loss_avg = train_epoch(
            dl_train, model, opt, loss_fn, device, show_progress
        )
        val_loss, val_loss_avg, pct_error_avg = validate(dl_val, model, loss_fn, device)
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_loss_avg": train_loss_avg,
                "val_loss_avg": val_loss_avg,
                "pct_error_avg": pct_error_avg,
            }
        )
    return metrics
