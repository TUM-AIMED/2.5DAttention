import sys
from enum import Enum
from pathlib import Path
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any, Optional

sys.path.insert(0, str(Path.cwd()))


@dataclass
class GeneralConfig:
    force_cpu: bool = False
    log_wandb: bool = False
    seed: int = 0
    compile: bool = False
    log_images: bool = True
    private_data: bool = True
    output_save_folder: Optional[Path] = None


class LoaderType(Enum):
    torch = 0
    monai = 1


@dataclass
class DataConfig:
    name: str = MISSING
    base_path: Optional[Path] = None
    loader: LoaderType = LoaderType.torch


class ModelTypes(Enum):
    twop5_pool = 0
    twop5_lstm = 1
    twop5_att = 2
    twop5_tf = 3
    acs_direct = 4
    acs_3d = 5
    acs_twop5 = 6


class DataViewAxis(Enum):
    all_sides = 0
    only_x = 1
    only_y = 2
    only_z = 3


@dataclass
class UnfreezeConfig:
    train_mode: int = -1
    feature_extractor: int = -1


@dataclass
class ModelConfig:
    type: ModelTypes = MISSING
    data_view_axis: DataViewAxis = DataViewAxis.all_sides
    backbone: str = MISSING
    in_channels: int = MISSING
    num_classes: int = MISSING
    feature_dim: Optional[int] = None
    unfreeze: UnfreezeConfig = field(default_factory=UnfreezeConfig)
    additional_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class HyperparamConfig:
    epochs: int = MISSING
    train_bs: int = MISSING
    val_bs: int = MISSING
    test_bs: int = MISSING
    grad_acc_steps: int = 1
    overfit: Optional[int] = None
    opt_args: dict[str, Any] = MISSING


class TransformLibrary(Enum):
    torchio = 0
    monai = 1


@dataclass
class TransformConfig:
    tf_library: TransformLibrary = TransformLibrary.torchio
    train_tf: dict[str, Any] = field(default_factory=dict)
    val_tf: dict[str, Any] = field(default_factory=dict)
    test_tf: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrivacyConfig:
    use_privacy: bool = False
    fix_model_for_privacy: bool = False
    epsilon: float = MISSING
    delta: float = MISSING
    clip_norm: float = MISSING
    accountant: str = "prv"


@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loader: dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    hyperparams: HyperparamConfig = field(default_factory=HyperparamConfig)
    wandb: dict[str, Any] = field(default_factory=dict)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


def load_config_store():
    configstore = ConfigStore.instance()
    configstore.store(name="base_config", node=Config)
