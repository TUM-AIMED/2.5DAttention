import torch
import wandb
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

sgmd = torch.nn.Softmax(dim=1)


def train_step(data, label, model, opt, loss_fn, settings):
    opt.zero_grad()
    data = data.to(**settings)
    pred, att_map = model(data)
    loss = loss_fn(pred, label.to(device=settings["device"]).squeeze())
    loss.backward()
    opt.step()
    return loss


def train(train_dl, model, opt, loss_fn, settings):
    pbar = tqdm(train_dl, total=len(train_dl), desc="Training", leave=False)
    for data, label in pbar:
        loss = train_step(data, label, model, opt, loss_fn, settings)
        pbar.set_description_str(f"Loss: {loss.item():.3f}")


def validate(val_dl, metric_fns, model, settings, add_wandb_plots=False):
    with torch.inference_mode():
        preds, labels = [], []
        for data, label in tqdm(
            val_dl, total=len(val_dl), desc="Validating", leave=False
        ):
            data = data.to(**settings)
            pred, att_map = model(data)
            pred = sgmd(pred)
            preds.append(pred.cpu())
            labels.append(label)
        preds, labels = torch.vstack(preds), torch.concatenate(labels).squeeze()
        metrics = {
            name: metric_fn(preds, labels).item()
            for name, metric_fn in metric_fns.items()
        }
        if add_wandb_plots:
            visualize_att_map(metrics, att_map, data)
    return metrics


def visualize_att_map(metrics, att_map, data):
    att_map_plot = att_map.cpu().numpy()[0][::-1]
    data_plot = data.cpu().numpy()[0, 14, 0, :, :]
    fig_width = 15
    fig_height = 5
    aspect_ratio = 1
    image_width = fig_width / (2 + aspect_ratio)
    bar_width = fig_width / (2 + aspect_ratio)
    third_image_width = aspect_ratio * image_width
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [image_width, bar_width, third_image_width]},
    )
    ax[0].imshow(data_plot, cmap="gray", aspect="auto")
    ax[1].barh(np.arange(att_map_plot.shape[0]), att_map_plot, align="edge")
    ax[1].set_ylim(0, att_map_plot.shape[0])
    ax[1].invert_yaxis()
    ax[2].imshow(
        data_plot
        * np.repeat(att_map_plot, data_plot.shape[-1]).reshape(
            *att_map_plot.shape,
            data_plot.shape[-1],
        ),
        cmap="gray",
        aspect="auto",
    )
    for i, a in enumerate(ax):
        if i > 0:
            a.set_yticklabels([])
            a.set_yticks([])
        a.set_xticklabels([])
        a.set_xticks([])
    ax[0].set_ylabel("Attention maps")
    fig.subplots_adjust(wspace=0)
    metrics["att_map"] = fig
    metrics["frontal_view"] = wandb.Image(data_plot)
    plt.close(fig)
