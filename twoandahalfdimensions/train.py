import sys
import hydra
import torch
import wandb
import torchmetrics
import seaborn as sn
from pathlib import Path
from tqdm import trange
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from numpy.random import seed as npseed
from random import seed as rseed


sys.path.insert(0, str(Path.cwd()))

from twoandahalfdimensions.utils.data import make_data
from twoandahalfdimensions.utils.training_utils import train, validate
from twoandahalfdimensions.utils.config import Config, load_config_store
from twoandahalfdimensions.models.preset_models import make_model_from_config

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
    model = torch.compile(model)
    opt = torch.optim.NAdam(model.parameters(), lr=config.hyperparams.lr)

    N_params_total = get_num_params(model)
    N_params_feature_extractor = get_num_params(model.feature_extractor)
    N_params_classifier = get_num_params(model.classifier)
    N_params_reduce = get_num_params(model.reduce_3d_module)
    if config.general.log_wandb:
        config_dict = OmegaConf.to_container(config)
        wandb.init(
            config=config_dict,
            settings=wandb.Settings(start_method="thread"),
            reinit=True,
            **config_dict["wandb"],
        )
        wandb.log(
            {
                "num_params": {
                    "total": N_params_total,
                    "feature_extractor": N_params_feature_extractor,
                    "classifier": N_params_classifier,
                    "2.5DModule": N_params_reduce,
                }
            }
        )
    else:
        print("Model parameters:")
        print(f"\tTotal: {N_params_total:,}")
        print(f"\tFeature Extractor: {N_params_feature_extractor:,}")
        print(f"\tClassifier: {N_params_classifier:,}")
        print(f"\t2.5D Module: {N_params_reduce:,}")

    train_metrics = validate(
        train_dl, metric_fns, model, settings, add_wandb_plots=config.general.log_wandb
    )
    val_metrics = validate(
        val_dl, metric_fns, model, settings, add_wandb_plots=config.general.log_wandb
    )
    if config.general.log_wandb:
        wandb.log({"train": train_metrics, "val": val_metrics, "epoch": 0})
    for epoch in trange(config.hyperparams.epochs, desc="Epochs", leave=False):
        train(train_dl, model, opt, loss_fn, settings)
        train_metrics = validate(
            train_dl,
            metric_fns,
            model,
            settings,
            add_wandb_plots=config.general.log_wandb,
        )
        val_metrics = validate(
            val_dl,
            metric_fns,
            model,
            settings,
            add_wandb_plots=config.general.log_wandb,
        )
        if config.general.log_wandb:
            wandb.log({"train": train_metrics, "val": val_metrics, "epoch": epoch + 1})
    test_metrics = validate(
        test_dl, metric_fns, model, settings, add_wandb_plots=config.general.log_wandb
    )
    if config.general.log_wandb:
        wandb.log({"test": test_metrics})


if __name__ == "__main__":
    main()
