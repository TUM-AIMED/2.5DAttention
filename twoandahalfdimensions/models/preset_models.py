from torch import nn, ones_like
from torchvision import models
from copy import deepcopy

from twoandahalfdimensions.models.twoandahalfdmodel import TwoAndAHalfDModel


def make_model_adaptions(feature_extractor, classifier, input_channels=3, freeze_feature_extractor=False)
    if input_channels != 3:
        scatter_conv = nn.Conv2d(input_channels, 3, 1, bias=False)
        scatter_conv.weight.set_(ones_like(scatter_conv.weight))
    if freeze_feature_extractor:
        for param in feature_extractor.parameters():
            param.requires_grad_(False)
    return TwoAndAHalfDModel(feature_extractor, classifier, classifier.weight.shape[1])

def make_resnet18(*args, **kwargs):
    feature_extractor = models.resnet18(models.ResNet18_Weights.DEFAULT)
    classifier = deepcopy(feature_extractor.fc)
    feature_extractor.fc = nn.Identity()
    return make_model_adaptions(feature_extractor, classifier, *args, **kwargs)






