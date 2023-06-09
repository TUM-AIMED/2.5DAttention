import hydra
import torch
import wandb
from pathlib import Path
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import torchmetrics

import sys

sys.path.insert(0, str(Path.cwd()))

from twoandahalfdimensions.utils.data import make_data

from twoandahalfdimensions.utils.config import Config, load_config_store
from twoandahalfdimensions.models.preset_models import make_model_from_config

sgmd = torch.nn.Softmax(dim=1)
load_config_store()


def train(train_dl, model, opt, loss_fn, settings):
    pbar = tqdm(train_dl, total=len(train_dl), desc="Training", leave=False)
    for data, label in pbar:
        opt.zero_grad()
        data = data.to(**settings)
        pred, att_map = model(data)
        loss = loss_fn(pred, label.to(device=settings["device"]).squeeze())
        loss.backward()
        opt.step()
        pbar.set_description_str(f"Loss: {loss.item():.3f}")


def validate(val_dl, metric_fns, model, settings):
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
    return metrics


@hydra.main(version_base=None, config_path=str(Path.cwd() / "configs"))
def main(config: Config):
    metric_fns = {
        "mcc": torchmetrics.MatthewsCorrCoef(task="multiclass", num_classes=11),
        "AUROC": torchmetrics.AUROC(task="multiclass", num_classes=11),
        "F1-Score": torchmetrics.F1Score(task="multiclass", num_classes=11),
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=11),
    }

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not config.general.force_cpu
        else torch.device("cpu")
    )
    settings = {"device": device, "dtype": torch.float32}
    train_ds, val_ds, test_ds = make_data(config)
    train_dl, val_dl, test_dl = (
        DataLoader(
            train_ds,
            batch_size=config.hyperparams.train_bs,
            shuffle=True,
            **config.loader,
        ),
        DataLoader(val_ds, batch_size=config.hyperparams.val_bs, **config.loader),
        DataLoader(test_ds, batch_size=config.hyperparams.test_bs, **config.loader),
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    model = make_model_from_config(config)

    model = model.to(**settings)
    opt = torch.optim.NAdam(model.parameters(), lr=config.hyperparams.lr)

    if config.general.log_wandb:
        config_dict = OmegaConf.to_container(config)
        wandb.init(
            config=config_dict,
            settings=wandb.Settings(start_method="thread"),
            reinit=True,
            **config_dict["wandb"],
        )

    train_metrics = validate(train_dl, metric_fns, model, settings)
    val_metrics = validate(val_dl, metric_fns, model, settings)
    if config.general.log_wandb:
        wandb.log({"train": train_metrics, "val": val_metrics, "epoch": 0})
    for epoch in trange(config.hyperparams.epochs, desc="Epochs", leave=False):
        train(train_dl, model, opt, loss_fn, settings)
        train_metrics = validate(train_dl, metric_fns, model, settings)
        val_metrics = validate(val_dl, metric_fns, model, settings)
        if config.general.log_wandb:
            wandb.log({"train": train_metrics, "val": val_metrics, "epoch": epoch + 1})
    test_metrics = validate(test_dl, metric_fns, model, settings)
    if config.general.log_wandb:
        wandb.log({"test": test_metrics})


if __name__ == "__main__":
    main()
