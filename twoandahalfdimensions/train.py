import sys
import hydra
import torch
import wandb
import torchmetrics
import seaborn as sn
from pathlib import Path
from tqdm import trange
from omegaconf import OmegaConf
from numpy import mean
from numpy.random import seed as npseed
from random import seed as rseed

sys.path.insert(0, str(Path.cwd()))

from twoandahalfdimensions.utils.data import make_data, make_loader
from twoandahalfdimensions.utils.training_utils import train, validate
from twoandahalfdimensions.utils.config import Config, load_config_store
from twoandahalfdimensions.models.preset_models import make_model_from_config
from twoandahalfdimensions.models.twoandahalfdmodel import TwoAndAHalfDModel
from datetime import datetime

sn.set_theme(
    context="notebook",
    style="white",
    font="Times New Roman",
    font_scale=1.2,
    palette="viridis",
)
sn.despine()
sn.set(rc={"figure.figsize": (12, 12)}, font_scale=1.2)
colors = {
    "red": "firebrick",
    "blue": "steelblue",
    "green": "forestgreen",
    "purple": "darkorchid",
    "orange": "darkorange",
    "gray": "lightslategray",
    "black": "black",
}

load_config_store()

get_num_params = lambda model: sum(
    p.numel() for p in model.parameters() if p.requires_grad
)


@hydra.main(version_base=None, config_path=str(Path.cwd() / "configs"))
def main(config: Config):
    torch.manual_seed(config.general.seed)
    npseed(config.general.seed)
    rseed(config.general.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    task = "multiclass" if config.model.num_classes > 1 else "binary"
    num_classes = config.model.num_classes if config.model.num_classes > 1 else None
    metric_fns = {
        "mcc": torchmetrics.MatthewsCorrCoef(task=task, num_classes=num_classes),
        "AUROC": torchmetrics.AUROC(task=task, num_classes=num_classes),
        "F1-Score": torchmetrics.F1Score(task=task, num_classes=num_classes),
        "accuracy": torchmetrics.Accuracy(task=task, num_classes=num_classes),
    }

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not config.general.force_cpu
        else torch.device("cpu")
    )
    settings = {"device": device, "dtype": torch.float32}
    train_ds, val_ds, test_ds = make_data(config)
    train_dl, val_dl, test_dl = make_loader(config, (train_ds, val_ds, test_ds))
    loss_fn = (
        torch.nn.CrossEntropyLoss()
        if config.model.num_classes > 1
        else torch.nn.BCEWithLogitsLoss()
    )
    activation_fn = (
        torch.nn.Softmax(dim=1) if config.model.num_classes > 1 else torch.nn.Sigmoid()
    )
    model = make_model_from_config(config)
    model.eval()

    model = model.to(**settings)
    if config.general.compile:
        model = torch.compile(model)
    opt = torch.optim.NAdam(model.parameters(), **config.hyperparams.opt_args)

    N_params_total = get_num_params(model)
    param_dict = {"num_params": {"total": N_params_total}}
    if isinstance(model, TwoAndAHalfDModel):
        N_params_feature_extractor = get_num_params(model.feature_extractor)
        N_params_classifier = get_num_params(model.classifier)
        N_params_reduce = get_num_params(model.reduce_3d_module)
        param_dict["num_params"].update(
            {
                "feature_extractor": N_params_feature_extractor,
                "classifier": N_params_classifier,
                "2.5DModule": N_params_reduce,
            }
        )
    if config.general.log_wandb:
        config_dict = OmegaConf.to_container(config)
        wandb.init(
            config=config_dict,
            settings=wandb.Settings(start_method="thread"),
            reinit=True,
            **config_dict["wandb"],
        )
        wandb.log(param_dict)
    else:
        print("Model parameters:")
        for key, val in param_dict["num_params"].items():
            print(f"\t{key}: {val:,}")
    data_view_axes = (
        config.model.data_view_axis
        if config.general.log_wandb and config.general.log_images
        else None
    )

    train_metrics = validate(
        config,
        train_dl,
        metric_fns,
        model,
        settings,
        activation_fn=activation_fn,
        data_vis_axes=data_view_axes,
        logging_step=0,
    )
    val_metrics = validate(
        config,
        val_dl,
        metric_fns,
        model,
        settings,
        activation_fn=activation_fn,
        data_vis_axes=data_view_axes,
        logging_step=0,
    )
    if config.general.log_wandb:
        wandb.log({"train": train_metrics, "val": val_metrics, "epoch": 0})
    for epoch in trange(config.hyperparams.epochs, desc="Epochs", leave=False):
        epoch_logging_dict = {}
        if epoch > config.model.unfreeze.train_mode:
            model.train()
        losses = train(
            train_dl,
            model,
            opt,
            loss_fn,
            settings,
            grad_acc_steps=config.hyperparams.grad_acc_steps,
        )
        model.eval()
        if epoch == config.model.unfreeze.feature_extractor:
            for param in model.parameters():
                param.requires_grad_(True)
            if config.general.log_wandb:
                param_dict = {"total": get_num_params(model)}
                if isinstance(model, TwoAndAHalfDModel):
                    param_dict.update(
                        {"feature_extractor": get_num_params(model.feature_extractor)}
                    )
                epoch_logging_dict["num_params"] = param_dict
        if config.general.log_wandb:
            epoch_logging_dict["train"] = validate(
                config,
                train_dl,
                metric_fns,
                model,
                settings,
                activation_fn=activation_fn,
                data_vis_axes=data_view_axes,
                logging_step=epoch + 1,
            )
            epoch_logging_dict["train"]["loss"] = mean(losses)
            epoch_logging_dict["val"] = validate(
                config,
                val_dl,
                metric_fns,
                model,
                settings,
                activation_fn=activation_fn,
                data_vis_axes=data_view_axes,
                logging_step=epoch + 1,
            )
            epoch_logging_dict["epoch"] = epoch + 1
            wandb.log(epoch_logging_dict)
    test_metrics = validate(
        config,
        test_dl,
        metric_fns,
        model,
        settings,
        activation_fn=activation_fn,
        data_vis_axes=data_view_axes,
        logging_step=epoch + 1,
    )
    if config.general.log_wandb:
        wandb.log({"test": test_metrics})
    if config.general.output_save_folder:
        if config.general.log_wandb:
            save_dir: Path = (
                config.general.output_save_folder / f"{wandb.run.name}_{wandb.run.id}"
            )
        else:
            save_dir: Path = (
                config.general.output_save_folder
                / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
        save_dir.mkdir(exist_ok=True, parents=False)
        torch.save(
            {"config": config, "model_weights": model.state_dict()},
            save_dir / "final_state.pt",
        )


if __name__ == "__main__":
    main()
