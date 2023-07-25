import torch
import wandb
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import Optional

from twoandahalfdimensions.utils.config import DataViewAxis, Config


def train_step(data, label, model, opt, loss_fn, settings, do_step=True):
    data = data.to(**settings)
    pred, att_map = model(data)
    loss = loss_fn(pred.squeeze(), label.to(device=settings["device"]).squeeze())
    loss.backward()
    if do_step:
        opt.step()
        opt.zero_grad()
    return loss.detach().item()


def train(train_dl, model, opt, loss_fn, settings, grad_acc_steps=1):
    pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc="Training", leave=False)
    losses = []
    for i, (data, label) in pbar:
        loss = train_step(
            data,
            label,
            model,
            opt,
            loss_fn,
            settings,
            do_step=((i + 1) % grad_acc_steps) == 0,
        )
        losses.append(loss)
        pbar.set_description_str(f"Loss: {loss:.3f}")
    return losses


def validate(
    config: Config,
    val_dl,
    metric_fns,
    model,
    settings,
    activation_fn: torch.nn.Module,
    logging_step: int,
    data_vis_axes: Optional[DataViewAxis] = None,
):
    with torch.inference_mode():
        preds, labels = [], []
        for data, label in tqdm(
            val_dl, total=len(val_dl), desc="Validating", leave=False
        ):
            data = data.to(**settings)
            pred, att_map = model(data)
            pred = activation_fn(pred)
            preds.append(pred.cpu())
            labels.append(label)
        preds, labels = (
            torch.vstack(preds).squeeze(),
            torch.concatenate(labels).squeeze(),
        )
        if preds is not torch.Tensor:
            preds = torch.Tensor(preds)
        if labels is not torch.Tensor:
            labels = torch.Tensor(labels)
        metrics = {
            name: metric_fn(preds, labels).item()
            for name, metric_fn in metric_fns.items()
        }
        if data_vis_axes and att_map is not None:
            fig = visualize_att_map(att_map, data, data_vis_axes)
            if config.general.private_data:
                save_dir = (
                    config.general.output_save_folder
                    / f"{wandb.run.name}_{wandb.run.id}"
                )
                save_dir.mkdir(exist_ok=True)
                fig.savefig(save_dir / f"{logging_step}_att_map.svg")

            else:
                metrics["att_map"] = fig
                # metrics["frontal_view"] = wandb.Image(data_plot)
            plt.close(fig)
    return metrics


def visualize_att_map(att_map, data, data_vis_axes: DataViewAxis):
    BATCH_INDEX = -1
    att_map_numpy = att_map.cpu().numpy()[BATCH_INDEX]  # [::-1]
    data_numpy = data.cpu().numpy()[BATCH_INDEX]  # (C, Z, X, Y)
    C, N_Z, N_X, N_Y = data_numpy.shape
    att_idx = [0, N_Z, N_Z + N_X, N_Z + N_X + N_Y]
    view_axes: list[int]
    match data_vis_axes:
        case DataViewAxis.all_sides:
            view_axes = [1, 2, 3]
        case DataViewAxis.only_z:
            view_axes = [1]
        case DataViewAxis.only_x:
            view_axes = [2]
        case DataViewAxis.only_y:
            view_axes = [3]
        case _:
            raise ValueError(f"Not supported")
    fig_width = 15
    fig_height = 5 * len(view_axes)
    aspect_ratio = 1
    image_width = fig_width / (2 + aspect_ratio)
    bar_width = fig_width / (2 + aspect_ratio)
    third_image_width = aspect_ratio * image_width
    fig, axs = plt.subplots(
        len(view_axes),
        3,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [image_width, bar_width, third_image_width]},
    )
    plt_kwargs = {"aspect": "auto"}
    axes_names = ("X-Axis", "Y-Axis", "Z-Axis")
    if C == 1:
        plt_kwargs["cmap"] = "gray"
    for j, view_axis in enumerate(view_axes):
        if len(view_axes) > 1:
            ax = axs[j]
            att_map_plot = att_map_numpy[att_idx[j] : att_idx[j + 1]][::-1]
        else:
            ax = axs
            att_map_plot = att_map_numpy[::-1]
        if view_axis == 1:  # Z
            data_plot = data_numpy.copy().transpose(0, 2, 1, 3)  # C, X, Z, Y
            data_plot = data_plot[..., data_plot.shape[-1] // 2].squeeze()
            axis_order = (0, 2, 1)
        elif view_axis == 2:  # X
            data_plot = data_numpy.copy()  # C, Z, X, Y
            data_plot = data_plot[..., data_plot.shape[-1] // 2].squeeze()
            axis_order = (2, 0, 1)
        elif view_axis == 3:  # Y
            data_plot = data_numpy.copy().transpose(0, 2, 3, 1)  # C, X, Y, Z
            data_plot = data_plot[..., data_plot.shape[-1] // 2].squeeze()
            axis_order = (0, 1, 2)
        else:
            raise ValueError(f"More than 3 view axes not supported")
        data_index: list = [slice(None) for _ in data_numpy.shape]
        data_index[view_axis] = att_map_plot.argmax()
        att_slice = data_numpy[tuple(data_index)].squeeze()
        ax[0].imshow(data_plot.T, **plt_kwargs)
        ax[1].barh(np.arange(att_map_plot.shape[0]), att_map_plot, align="edge")
        ax[1].set_xlim(0, att_map_numpy.max())
        ax[1].set_ylim(0, att_map_plot.shape[0])
        ax[1].invert_yaxis()
        ax[2].imshow(att_slice.T, **plt_kwargs)
        for i, a in enumerate(ax):
            if i > 0:
                a.set_yticklabels([])
                a.set_yticks([])
            a.set_xticklabels([])
            a.set_xticks([])
        ax[0].set_ylabel(f"Attention maps {axes_names[axis_order[1]]}")
        ax[0].set_xlabel(
            f"{axes_names[axis_order[0]]} (Middle of {axes_names[axis_order[2]]})"
        )
        ax[1].set_xlabel(f"Attention along {axes_names[axis_order[1]]}")
        ax[2].set_xlabel(f"Highest attenuated slice along {axes_names[axis_order[1]]}")
        ax[0].grid(False)
        ax[1].grid(True)
        ax[2].grid(False)
        fig.subplots_adjust(wspace=0)
    return fig
