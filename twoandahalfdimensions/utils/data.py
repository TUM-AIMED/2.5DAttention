import medmnist
import torchio as tio
from numpy import float32
from typing import Any
from torch.utils.data import DataLoader
from monai.data.dataloader import DataLoader as MDTL
from monai import transforms as mtf

from twoandahalfdimensions.utils.config import Config, LoaderType, TransformLibrary
from twoandahalfdimensions.utils.CTImageFolder import CTImageFolder

idty = lambda x: x


def dict_from_module(module):
    context = {}
    for setting in dir(module):
        if setting.isalpha():
            context[setting] = getattr(module, setting)
    return context


def make_transforms(config: Config, tf_dict: dict[str, Any]):
    match config.transforms.tf_library:
        case TransformLibrary.torchio:
            tf_module_dict = dict_from_module(tio)
        case TransformLibrary.monai:
            tf_module_dict = dict_from_module(mtf)
        case default:
            raise ValueError(f"{default} transform library not supported")
    list_tfs = [
        tf_module_dict[tf](**kwargs) if kwargs is not None else tf_module_dict[tf]()
        for tf, kwargs in tf_dict.items()
    ]
    return tf_module_dict["Compose"](list_tfs)


def overfit_mnist(config, train_ds, val_ds, test_ds):
    if config.hyperparams.overfit:
        train_ds.imgs = train_ds.imgs[: config.hyperparams.overfit]
        train_ds.labels = train_ds.labels[: config.hyperparams.overfit]
        val_ds.imgs, val_ds.labels = train_ds.imgs, train_ds.labels
        test_ds.imgs, test_ds.labels = train_ds.imgs, train_ds.labels


def make_data(config: Config):
    transforms = [
        make_transforms(config, tf) if len(tf) > 0 else None
        for tf in (
            config.transforms.train_tf,
            config.transforms.val_tf,
            config.transforms.test_tf,
        )
    ]
    label_transform = (
        None if config.model.num_classes > 1 else lambda x: x.astype(float32)
    )
    match config.data.name:
        case "ct_imagefolder":
            train_ds = CTImageFolder(
                config.data.base_path / "train",
                transform=transforms[0],
                label_transform=label_transform,
            )
            val_ds = CTImageFolder(
                config.data.base_path / "val",
                transform=transforms[1],
                label_transform=label_transform,
            )
            test_ds = CTImageFolder(
                config.data.base_path / "test",
                transform=transforms[2],
                label_transform=label_transform,
            )
        case "organmnist3d":
            train_ds, val_ds, test_ds = (
                medmnist.OrganMNIST3D(
                    split,
                    download=True,
                    root="./data",
                    transform=tf,
                    target_transform=label_transform,
                )
                for split, tf in zip(["train", "val", "test"], transforms)
            )
            overfit_mnist(config, train_ds, val_ds, test_ds)
        case "nodulemnist3d":
            train_ds, val_ds, test_ds = (
                medmnist.NoduleMNIST3D(
                    split,
                    download=True,
                    root="./data",
                    transform=tf,
                    target_transform=label_transform,
                )
                for split, tf in zip(["train", "val", "test"], transforms)
            )
            overfit_mnist(config, train_ds, val_ds, test_ds)
        case "fracturemnist3d":
            train_ds, val_ds, test_ds = (
                medmnist.FractureMNIST3D(
                    split,
                    download=True,
                    root="./data",
                    transform=tf,
                    target_transform=label_transform,
                )
                for split, tf in zip(["train", "val", "test"], transforms)
            )
            overfit_mnist(config, train_ds, val_ds, test_ds)
        case "adrenalmnist3d":
            train_ds, val_ds, test_ds = (
                medmnist.AdrenalMNIST3D(
                    split,
                    download=True,
                    root="./data",
                    transform=tf,
                    target_transform=label_transform,
                )
                for split, tf in zip(["train", "val", "test"], transforms)
            )
            overfit_mnist(config, train_ds, val_ds, test_ds)
        case "vesselmnist3d":
            train_ds, val_ds, test_ds = (
                medmnist.VesselMNIST3D(
                    split,
                    download=True,
                    root="./data",
                    transform=tf,
                    target_transform=label_transform,
                )
                for split, tf in zip(["train", "val", "test"], transforms)
            )
            overfit_mnist(config, train_ds, val_ds, test_ds)
        case "synapsemnist3d":
            train_ds, val_ds, test_ds = (
                medmnist.SynapseMNIST3D(
                    split,
                    download=True,
                    root="./data",
                    transform=tf,
                    target_transform=label_transform,
                )
                for split, tf in zip(["train", "val", "test"], transforms)
            )
            overfit_mnist(config, train_ds, val_ds, test_ds)
        case _:
            raise ValueError(f"{config.data.name} not supported")
    return train_ds, val_ds, test_ds


def make_loader(config: Config, datasets: tuple):
    train_ds, val_ds, test_ds = datasets
    match config.data.loader:
        case LoaderType.torch:
            dl_class = DataLoader
        case LoaderType.monai:
            dl_class = MDTL
        case default:
            raise ValueError(f"LoaderType {default} not supported")

    train_dl, val_dl, test_dl = (
        dl_class(
            train_ds,
            batch_size=config.hyperparams.train_bs,
            shuffle=True,
            **config.loader,
        ),
        dl_class(val_ds, batch_size=config.hyperparams.val_bs, **config.loader),
        dl_class(test_ds, batch_size=config.hyperparams.test_bs, **config.loader),
    )
    return train_dl, val_dl, test_dl
