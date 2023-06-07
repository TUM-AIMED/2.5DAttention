import torch.nn as nn


class TwoAndAHalfDModel(nn.Module):
    def __init__(
        self, feature_extractor: nn.Module, classifier: nn.Module, hidden_size: int
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.att = nn.MultiheadAttention(hidden_size, 8)
        self.classifier = classifier

    def forward(self, x):
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time

        query = features.mean(0, keepdims=True)
        features, att_map = self.att(query, features, features)
        out = self.classifier(features.squeeze(0))
        return out, att_map
