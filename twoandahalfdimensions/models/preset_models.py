from torch import nn, ones_like
from torchvision import models

from twoandahalfdimensions.models.twoandahalfdmodel import (
    TwoAndAHalfDAttention,
    TwoAndAHalfDLSTM,
    TwoAndAHalfDTransformer,
    TwoAndAHalfDPool,
)
from twoandahalfdimensions.utils.config import Config, ModelTypes
from acsconv import converters


class FakeAttOutput(nn.Module):
    def __init__(self, model: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model: nn.Module = model

    def forward(self, x):
        return self.model(x), None


def is_our_model(type: ModelTypes) -> bool:
    return type in [
        ModelTypes.twop5_pool,
        ModelTypes.twop5_lstm,
        ModelTypes.twop5_att,
        ModelTypes.twop5_tf,
    ]


def make_model_from_config(config: Config):
    our_models: bool = is_our_model(config.model.type)
    if "resnet" in config.model.backbone:
        (
            feature_extractor,
            classifier,
            num_extracted_features,
            num_classified_features,
        ) = make_resnet(config, disassemble_model=our_models)
    elif "vit" in config.model.backbone:
        (
            feature_extractor,
            classifier,
            num_extracted_features,
            num_classified_features,
        ) = make_vit(config, disassemble_model=our_models)
    else:
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
        case ModelTypes.twop5_pool:
            model = TwoAndAHalfDPool
        case ModelTypes.twop5_lstm:
            model = TwoAndAHalfDLSTM
        case ModelTypes.twop5_att:
            model = TwoAndAHalfDAttention
        case ModelTypes.twop5_tf:
            model = TwoAndAHalfDTransformer
        case ModelTypes.acs_direct:
            converter = converters.ACSConverter
        case ModelTypes.acs_3d:
            converter = converters.Conv3dConverter
        case ModelTypes.acs_twop5:
            converter = converters.Conv2_5dConverter
        case default:
            raise ValueError(f"{default} not supported")
    if is_our_model(config.model.type):
        return model(
            feature_extractor,
            classifier,
            feature_size_in,
            feature_size_out,
            data_view_axis=config.model.data_view_axis,
            patchmode=config.model.patchmode,
            **config.model.additional_args,
        )
    else:
        model: nn.Module = converter(feature_extractor)
        model = FakeAttOutput(model.model)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in classifier.parameters():
            p.requires_grad_(True)
        return model


def make_resnet(config: Config, disassemble_model=True):
    match config.model.backbone:
        case "resnet18":
            feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        case "resnet34":
            feature_extractor = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        case "resnet50":
            feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        case "resnet101":
            feature_extractor = models.resnet101(
                weights=models.ResNet101_Weights.DEFAULT
            )
        case "resnet152":
            feature_extractor = models.resnet152(
                weights=models.ResNet152_Weights.DEFAULT
            )
        case _:
            raise ValueError(f"ResNet {config.model.backbone} not supported")
    num_extracted_features = feature_extractor.fc.weight.shape[1]
    classifier = nn.Linear(
        (
            num_extracted_features
            if config.model.feature_dim is None
            else config.model.feature_dim
        ),
        config.model.num_classes,
    )
    if disassemble_model:
        feature_extractor.fc = nn.Identity()
    else:
        feature_extractor.fc = classifier
    num_classified_features = classifier.weight.shape[1]
    return (
        feature_extractor,
        classifier,
        num_extracted_features,
        num_classified_features,
    )


def make_vit(config: Config, disassemble_model=True):
    match config.model.backbone:
        case "vit_b_16":
            feature_extractor = models.vit_b_16(models.ViT_B_16_Weights.DEFAULT)
        case "vit_b_32":
            feature_extractor = models.vit_b_32(models.ViT_B_32_Weights.DEFAULT)
        case "vit_l_16":
            feature_extractor = models.vit_l_16(models.ViT_L_16_Weights.DEFAULT)
        case "vit_l_32":
            feature_extractor = models.vit_l_32(models.ViT_L_32_Weights.DEFAULT)
        case "vit_h_14":
            feature_extractor = models.vit_h_14(models.ViT_H_14_Weights.DEFAULT)
    num_extracted_features = feature_extractor.heads[0].weight.shape[1]
    classifier = nn.Linear(
        (
            num_extracted_features
            if config.model.feature_dim is None
            else config.model.feature_dim
        ),
        config.model.num_classes,
    )
    if disassemble_model:
        feature_extractor.heads = nn.Identity()
    else:
        feature_extractor.heads = classifier
    num_classified_features = classifier.weight.shape[1]
    return (
        feature_extractor,
        classifier,
        num_extracted_features,
        num_classified_features,
    )
