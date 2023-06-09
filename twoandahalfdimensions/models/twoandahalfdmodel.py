from abc import ABC
from torch import Tensor, nn, ones
from typing import Optional


class TwoAndAHalfDModel(nn.Module, ABC):
    def __init__(
        self, feature_extractor: nn.Module, classifier: nn.Module, feature_size: int
    ) -> None:
        super().__init__()
        self.feature_extractor: nn.Module = feature_extractor
        self.classifier: nn.Module = classifier
        self.feature_size: int = feature_size
        self.reduce_3d_module: nn.Module


class TwoAndAHalfDAttention(TwoAndAHalfDModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        feature_size: int,
        num_heads: int = 8,
        **kwargs
    ):
        super().__init__(feature_extractor, classifier, feature_size)
        self.reduce_3d_module = nn.MultiheadAttention(
            self.feature_size, num_heads, batch_first=True, **kwargs
        )

    def forward(self, x: Tensor):
        N_scans, C, N_slices, px_x, px_y = x.shape
        x.transpose_(1, 2)
        x = x.reshape(N_scans * N_slices, C, px_x, px_y)
        features = self.feature_extractor(x)  # shape (N_scans*N_slices, num_classes)
        features = features.reshape(
            N_scans, N_slices, -1
        )  # shape (N_scans, N_slices, num_classes)

        query = features.mean(1, keepdims=True)
        features, att_map = self.reduce_3d_module(query, features, features)
        features.squeeze_(1)
        att_map.squeeze_(1)
        out = self.classifier(features)
        return out, att_map


class TwoAndAHalfDLSTM(TwoAndAHalfDModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        feature_size: int,
        hidden_size: Optional[int] = None,
        num_layers: int = 2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(feature_extractor, classifier, feature_size)
        self.reduce_3d_module = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size if hidden_size is not None else self.feature_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        N_scans, C, N_slices, px_x, px_y = x.shape
        x.transpose_(1, 2)
        x = x.reshape(N_scans * N_slices, C, px_x, px_y)
        features = self.feature_extractor(x)  # shape (N_scans*N_slices, num_classes)
        features = features.reshape(
            N_scans, N_slices, -1
        )  # shape (N_scans, N_slices, num_classes)
        summed_features, (h_n, c_n) = self.reduce_3d_module(features)
        summed_features = summed_features[:, -1]
        out = self.classifier(summed_features)
        return out, ones((N_scans, N_slices))
