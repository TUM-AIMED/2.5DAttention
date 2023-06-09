from torch import nn, ones_like, no_grad
from torchvision import models
from copy import deepcopy

from twoandahalfdimensions.models.twoandahalfdmodel import TwoAndAHalfDModel
from twoandahalfdimensions.utils.config import ModelTypes


def make_model_from_config(config):
    match config.model.type:
        case ModelTypes.twop5:
            match config.model.backbone:
                case "resnet18":
                    model = make_resnet18(
                        num_classes=config.model.num_classes,
                        input_channels=config.model.in_channels,
                        freeze_feature_extractor=config.model.freeze_feature_extractor,
                    )
                case _:
                    raise ValueError(f"Model {config.model.backbone} not supported yet")
        case _:
            raise ValueError(f"Type {config.model.type} not supported yet")
    return model


def make_model_adaptions(
    feature_extractor, classifier, input_channels=3, freeze_feature_extractor=False
):
    if input_channels != 3:
        scatter_conv = nn.Conv2d(input_channels, 3, 1, bias=False)
        for p in scatter_conv.parameters():
            p.requires_grad_(False)
        scatter_conv.weight.set_(ones_like(scatter_conv.weight))
    feature_extractor = nn.Sequential(scatter_conv, feature_extractor)
    if freeze_feature_extractor:
        for param in feature_extractor.parameters():
            param.requires_grad_(False)
    return TwoAndAHalfDModel(feature_extractor, classifier, classifier.weight.shape[1])

    feature_extractor = nn.Sequential([scatter_conv, feature_extractor])


def make_resnet18(num_classes: int, *args, **kwargs):
    feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if num_classes == 1000:
        classifier = deepcopy(feature_extractor.fc)
    else:
        classifier = nn.Linear(feature_extractor.fc.weight.shape[1], num_classes)
    feature_extractor.fc = nn.Identity()
    return make_model_adaptions(feature_extractor, classifier, *args, **kwargs)
