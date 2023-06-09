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


@dataclass
class DataConfig:
    name: str = MISSING


class ModelTypes(Enum):
    twop5_att = 0
    twop5_lstm = 1


@dataclass
class ModelConfig:
    type: ModelTypes = MISSING
    backbone: str = MISSING
    in_channels: int = MISSING
    num_classes: int = MISSING
    freeze_feature_extractor: bool = MISSING
    additional_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class HyperparamConfig:
    epochs: int = MISSING
    train_bs: int = MISSING
    val_bs: int = MISSING
    test_bs: int = MISSING
    lr: float = MISSING
    overfit: Optional[int] = None


@dataclass
class TransformConfig:
    train_tf: dict[str, Any] = field(default_factory=dict)
    val_tf: dict[str, Any] = field(default_factory=dict)
    test_tf: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loader: dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    hyperparams: HyperparamConfig = field(default_factory=HyperparamConfig)
    wandb: dict[str, Any] = field(default_factory=dict)
    transforms: TransformConfig = field(default_factory=TransformConfig)


def load_config_store():
    configstore = ConfigStore.instance()
    configstore.store(name="base_config", node=Config)
