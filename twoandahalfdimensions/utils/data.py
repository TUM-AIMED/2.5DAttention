import medmnist
import torchio as tio
from numpy import float32
from typing import Any
from twoandahalfdimensions.utils.config import Config

idty = lambda x: x


def make_transforms(tf_dict: dict[str, Any]):
    list_tfs = [getattr(tio, tf)(**kwargs) for tf, kwargs in tf_dict.items()]
    return tio.Compose(list_tfs)


def overfit_mnist(config, train_ds, val_ds, test_ds):
    if config.hyperparams.overfit:
        train_ds.imgs = train_ds.imgs[: config.hyperparams.overfit]
        train_ds.labels = train_ds.labels[: config.hyperparams.overfit]
        val_ds.imgs, val_ds.labels = train_ds.imgs, train_ds.labels
        test_ds.imgs, test_ds.labels = train_ds.imgs, train_ds.labels


def make_data(config: Config):
    transforms = [
        make_transforms(tf) if len(tf) > 0 else None
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
