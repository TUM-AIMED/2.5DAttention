import torchio as tio
from typing import Any
from medmnist import OrganMNIST3D
from twoandahalfdimensions.utils.config import Config

idty = lambda x: x


def make_transforms(tf_dict: dict[str, Any]):
    list_tfs = [getattr(tio, tf)(**kwargs) for tf, kwargs in tf_dict.items()]
    return tio.Compose(list_tfs)


def make_data(config: Config):
    transforms = [
        make_transforms(tf) if len(tf) > 0 else idty
        for tf in (
            config.transforms.train_tf,
            config.transforms.val_tf,
            config.transforms.test_tf,
        )
    ]

    if config.data.name == "organmnist3d":
        train_ds, val_ds, test_ds = (
            OrganMNIST3D(split, download=True, root="./data", transform=tf)
            for split, tf in zip(["train", "val", "test"], transforms)
        )
        if config.hyperparams.overfit:
            train_ds.imgs = train_ds.imgs[: config.hyperparams.overfit]
            train_ds.labels = train_ds.labels[: config.hyperparams.overfit]
            val_ds = test_ds = train_ds
    else:
        raise ValueError(f"{config.data.name} not supported")
    return train_ds, val_ds, test_ds
