from torch import nn, ones_like, no_grad
from torchvision import models
from copy import deepcopy

from twoandahalfdimensions.models.twoandahalfdmodel import (
    TwoAndAHalfDAttention,
    TwoAndAHalfDLSTM,
)
from twoandahalfdimensions.utils.config import Config, ModelTypes


def make_model_from_config(config: Config):
    match config.model.backbone:
        case "resnet18":
            (
                feature_extractor,
                classifier,
                num_extracted_features,
                num_classified_features,
            ) = make_resnet18(config)
        case _:
            raise ValueError(f"Model {config.model.backbone} not supported yet")
    model = make_model_adaptions(
        config,
        feature_extractor,
        classifier,
        num_extracted_features,
        num_classified_features,
    )
    return model


def make_model_adaptions(
    config: Config,
    feature_extractor: nn.Module,
    classifier: nn.Module,
    feature_size_in: int,
    feature_size_out: int,
):
    if config.model.in_channels != 3:
        scatter_conv = nn.Conv2d(config.model.in_channels, 3, 1, bias=False)
        for p in scatter_conv.parameters():
            p.requires_grad_(False)
        scatter_conv.weight.set_(ones_like(scatter_conv.weight))
    feature_extractor = nn.Sequential(scatter_conv, feature_extractor)
    for param in feature_extractor.parameters():
        param.requires_grad_(False)

    match config.model.type:
        case ModelTypes.twop5_att:
            model = TwoAndAHalfDAttention
        case ModelTypes.twop5_lstm:
            model = TwoAndAHalfDLSTM
        case _:
            raise ValueError(f"Type {config.model.type} not supported yet")
    return model(
        feature_extractor,
        classifier,
        feature_size_in,
        feature_size_out,
        data_view_axis=config.model.data_view_axis,
        **config.model.additional_args,
    )


def make_resnet18(config: Config):
    feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_extracted_features = feature_extractor.fc.weight.shape[1]
    if config.model.num_classes == 1000:
        classifier = deepcopy(feature_extractor.fc)
    else:
        classifier = nn.Linear(
            num_extracted_features
            if config.model.feature_dim is None
            else config.model.feature_dim,
            config.model.num_classes,
        )
    feature_extractor.fc = nn.Identity()
    num_classified_features = classifier.weight.shape[1]
    return (
        feature_extractor,
        classifier,
        num_extracted_features,
        num_classified_features,
    )
