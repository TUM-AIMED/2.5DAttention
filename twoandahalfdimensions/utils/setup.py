import torch
import torchmetrics
from numpy.random import seed as npseed
from random import seed as rseed


def setup(config):
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
    activation_fn = (
        torch.nn.Softmax(dim=1) if config.model.num_classes > 1 else torch.nn.Sigmoid()
    )
    return metric_fns, settings, activation_fn
