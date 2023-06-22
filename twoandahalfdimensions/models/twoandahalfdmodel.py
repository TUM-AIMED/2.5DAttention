from abc import ABC
from torch import Tensor, nn, stack, concatenate


from twoandahalfdimensions.utils.config import DataViewAxis


class TwoAndAHalfDModel(nn.Module, ABC):
    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        feature_size_in: int,
        feature_size_out: int,
        data_view_axis: DataViewAxis,
    ) -> None:
        super().__init__()
        self.feature_extractor: nn.Module = feature_extractor
        self.classifier: nn.Module = classifier
        self.feature_size_in: int = feature_size_in
        self.feature_size_out: int = feature_size_out
        self.reduce_3d_module: nn.Module
        self.data_view_axis: DataViewAxis = data_view_axis

    def forward(self, x):
        N_scans, C, N_z, N_x, N_y = x.shape
        feature_list: list[Tensor] = []
        if self.data_view_axis in [DataViewAxis.all_sides, DataViewAxis.only_z]:
            feature_list.append(
                self.feature_extractor(
                    x.permute(0, 2, 1, 3, 4).reshape(N_scans * N_z, C, N_x, N_y)
                ).reshape(N_scans, N_z, -1)
            )
        if self.data_view_axis in [DataViewAxis.all_sides, DataViewAxis.only_x]:
            feature_list.append(
                self.feature_extractor(
                    x.permute(0, 3, 1, 2, 4).reshape(N_scans * N_x, C, N_z, N_y)
                ).reshape(N_scans, N_x, -1)
            )
        if self.data_view_axis in [DataViewAxis.all_sides, DataViewAxis.only_y]:
            feature_list.append(
                self.feature_extractor(
                    x.permute(0, 4, 1, 2, 3).reshape(N_scans * N_y, C, N_z, N_x)
                ).reshape(N_scans, N_y, -1)
            )
        features: Tensor = concatenate(
            feature_list, dim=1
        )  # shape (N_scans, N_slices* view_axes, num_features)
        return features


class TwoAndAHalfDAttention(TwoAndAHalfDModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        feature_size_in: int,
        feature_size_out: int,
        data_view_axis: DataViewAxis,
        num_heads: int = 8,
        **kwargs,
    ):
        super().__init__(
            feature_extractor,
            classifier,
            feature_size_in,
            feature_size_out,
            data_view_axis,
        )
        self.reduce_3d_module = nn.MultiheadAttention(
            self.feature_size_out, num_heads, batch_first=True, **kwargs
        )
        if feature_size_in != feature_size_out:
            self.feature_extractor = nn.Sequential(
                self.feature_extractor,
                nn.Linear(self.feature_size_in, self.feature_size_out),
            )

    def forward(self, x: Tensor):
        features = super().forward(x)
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
        feature_size_in: int,
        feature_size_out: int,
        data_view_axis: DataViewAxis,
        num_layers: int = 2,
        bidirectional: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            feature_extractor,
            classifier,
            feature_size_in,
            feature_size_out,
            data_view_axis,
        )
        self.reduce_3d_module = nn.LSTM(
            input_size=self.feature_size_in,
            hidden_size=self.feature_size_out,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            **kwargs,
        )

    def forward(self, x):
        features = super().forward(
            x
        )  # shape (N_scans, N_slices* view_axes, num_features)
        summed_features, (h_n, c_n) = self.reduce_3d_module(features)
        summed_features = summed_features[:, -1]
        out = self.classifier(summed_features)
        return out, None


class TwoAndAHalfDTransformer(TwoAndAHalfDModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        feature_size_in: int,
        feature_size_out: int,
        data_view_axis: DataViewAxis,
        **transformer_kwargs,
    ) -> None:
        super().__init__(
            feature_extractor,
            classifier,
            feature_size_in,
            feature_size_out,
            data_view_axis,
        )
        self.reduce_3d_module = nn.Transformer(
            self.feature_size_out, batch_first=True, **transformer_kwargs
        )
        if feature_size_in != feature_size_out:
            self.feature_extractor = nn.Sequential(
                self.feature_extractor,
                nn.Linear(self.feature_size_in, self.feature_size_out),
            )

    def forward(self, x: Tensor):
        features = super().forward(x)
        query = features.mean(1, keepdims=True)
        features = self.reduce_3d_module(features, query)
        features.squeeze_(1)
        out = self.classifier(features)
        return out, None
