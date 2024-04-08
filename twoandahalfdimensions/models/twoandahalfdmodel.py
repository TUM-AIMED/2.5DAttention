from abc import ABC
from torch import Tensor, nn, stack, concatenate

from einops import rearrange


from twoandahalfdimensions.utils.config import DataViewAxis, PatchPoolingMode


class Lambda(nn.Module):
    def __init__(self, lambda_expr) -> None:
        super().__init__()
        self.lambda_expr = lambda_expr

    def forward(self, x):
        return self.lambda_expr(x)


class TwoAndAHalfDModel(nn.Module, ABC):
    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        feature_size_in: int,
        feature_size_out: int,
        data_view_axis: DataViewAxis,
        patchify: bool,
        num_patches: int,
        patchmode: PatchPoolingMode = PatchPoolingMode.twostage,
    ) -> None:
        super().__init__()
        self.feature_extractor: nn.Module = feature_extractor
        self.classifier: nn.Module = classifier
        self.feature_size_in: int = feature_size_in
        self.feature_size_out: int = feature_size_out
        self.reduce_3d_module: nn.Module
        self.data_view_axis: DataViewAxis = data_view_axis

        self.patchify: bool = patchify
        self.num_patches: int = num_patches
        self.patchmode: PatchPoolingMode = patchmode

    def forward(self, x):
        N_scans, C, N_z, N_x, N_y = x.shape
        feature_list: list[Tensor] = []
        if self.data_view_axis in [DataViewAxis.all_sides, DataViewAxis.only_z]:
            reshaped_input = rearrange(x, "N C Z X Y -> (N Z) C X Y")
            if self.patchify:

                if N_x % self.num_patches != 0 or N_y % self.num_patches != 0:
                    raise ValueError(
                        f"Image of size {N_x}x{N_y} not divisible by {self.num_patches} patches"
                    )
                px, py = self.num_patches, self.num_patches

                reshaped_input = rearrange(
                    reshaped_input, "N C (px X) (py Y) -> (N px py) C X Y", px=px, py=py
                )
            features = self.feature_extractor(reshaped_input)
            if self.patchify:
                features = rearrange(
                    features,
                    "(N Z PX PY) F -> N Z (PX PY) F",
                    N=N_scans,
                    Z=N_z,
                    PX=px,
                    PY=py,
                )
            else:
                features = rearrange(features, "(N Z) F -> N Z F", N=N_scans, Z=N_z)
            feature_list.append(features)
        if self.data_view_axis in [DataViewAxis.all_sides, DataViewAxis.only_x]:
            reshaped_input = rearrange(x, "N C Z X Y -> (N X) C Z Y")
            if self.patchify:

                if N_y % self.num_patches != 0 or N_z % self.num_patches != 0:
                    raise ValueError(
                        f"Image of size {N_y}x{N_z} not divisible by {self.num_patches} patches"
                    )
                pz, py = self.num_patches, self.num_patches

                reshaped_input = rearrange(
                    reshaped_input, "N C (pz Z) (py Y) -> (N pz py) C Z Y", pz=pz, py=py
                )
            features = self.feature_extractor(reshaped_input)
            if self.patchify:
                features = rearrange(
                    features,
                    "(N X PZ PY) F -> N X (PZ PY) F",
                    N=N_scans,
                    X=N_x,
                    PZ=pz,
                    PY=py,
                )
            else:
                features = rearrange(features, "(N X) F -> N X F", N=N_scans, X=N_x)
            feature_list.append(features)
        if self.data_view_axis in [DataViewAxis.all_sides, DataViewAxis.only_y]:
            reshaped_input = rearrange(x, "N C Z X Y -> (N Y) C Z X")
            if self.patchify:

                if N_x % self.num_patches != 0 or N_z % self.num_patches != 0:
                    raise ValueError(
                        f"Image of size {N_x}x{N_z} not divisible by {self.num_patches} patches"
                    )
                px, pz = self.num_patches, self.num_patches

                reshaped_input = rearrange(
                    reshaped_input, "N C (pz Z) (px X) -> (N pz px) C Z X", pz=pz, px=px
                )

            features = self.feature_extractor(reshaped_input)
            if self.patchify:
                features = rearrange(
                    features,
                    "(N Y PZ PX) F -> N Y (PZ PX) F",
                    N=N_scans,
                    Y=N_y,
                    PZ=pz,
                    PX=px,
                )
            else:
                features = rearrange(features, "(N Y) F -> N Y F", N=N_scans, Y=N_y)

            feature_list.append(features)

        # note that when using patches *and* multiple axes
        # the input must have the same size in X,Y,Z
        # otherwise this will not work
        features: Tensor = concatenate(feature_list, dim=1)
        return features


class TwoAndAHalfDPool(TwoAndAHalfDModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        feature_size_in: int,
        feature_size_out: int,
        data_view_axis: DataViewAxis,
        patchify: bool,
        num_patches: int,
        patchmode: PatchPoolingMode = PatchPoolingMode.twostage,
        mode: str = "max",
    ) -> None:
        if patchify:
            raise ValueError("Patches not yet supported for pooling")
        super().__init__(
            feature_extractor,
            classifier,
            feature_size_in,
            feature_size_out,
            data_view_axis,
            patchify=patchify,
            num_patches=num_patches,
            patchmode=patchmode,
        )
        assert mode in ["max", "avg"], "only max and average supported"
        if mode == "max":
            self.reduce_3d_module = Lambda(lambda x: x.max(1)[0])
        elif mode == "avg":
            self.reduce_3d_module = Lambda(lambda x: x.mean(1))
        else:
            raise ValueError(f"{mode} not supported")

    def forward(self, x):
        x = super().forward(x)
        x = self.reduce_3d_module(x)
        x = self.classifier(x)
        return x, None


class TwoAndAHalfDAttention(TwoAndAHalfDModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        feature_size_in: int,
        feature_size_out: int,
        data_view_axis: DataViewAxis,
        patchify: bool,
        num_patches: int,
        patchmode: PatchPoolingMode = PatchPoolingMode.twostage,
        num_heads: int = 8,
        **kwargs,
    ):
        super().__init__(
            feature_extractor,
            classifier,
            feature_size_in,
            feature_size_out,
            data_view_axis,
            patchify=patchify,
            num_patches=num_patches,
            patchmode=patchmode,
        )
        if self.patchify and self.patchmode == PatchPoolingMode.twostage:
            self.reduce_3d_module = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        self.feature_size_out, num_heads, batch_first=True, **kwargs
                    ),
                    nn.MultiheadAttention(
                        self.feature_size_out, num_heads, batch_first=True, **kwargs
                    ),
                ]
            )
        else:
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
        if self.patchify:
            # we get a feature list of shape B slices patch patch F
            match self.patchmode:
                case PatchPoolingMode.onestage:
                    features = rearrange(features, "N S P F -> N (S P) F")
                    features, att_map = self.att_pool(features, self.reduce_3d_module)
                case PatchPoolingMode.twostage:
                    # first pool patches per slice then pool slices
                    N, S, _, F = features.shape
                    features = rearrange(features, "N S P F -> (N S) P F")
                    features, att_map = self.att_pool(
                        features, self.reduce_3d_module[0]
                    )  # slice features
                    features = rearrange(features, "(N S) F -> N S F", N=N, S=S, F=F)
                    features, att_map2 = self.att_pool(
                        features, self.reduce_3d_module[1]
                    )
                    att_map = [att_map, att_map2]
                case _:
                    raise ValueError("Not Supported")
        else:
            features, att_map = self.att_pool(features, self.reduce_3d_module)
        out = self.classifier(features)
        return out, att_map

    def att_pool(self, features, att_pool_layer):
        query = features.mean(1, keepdims=True)
        features, att_map = att_pool_layer(query, features, features)
        features.squeeze_(1)
        att_map.squeeze_(1)
        return features, att_map


class TwoAndAHalfDLSTM(TwoAndAHalfDModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        feature_size_in: int,
        feature_size_out: int,
        data_view_axis: DataViewAxis,
        patchify: bool,
        num_patches: int,
        patchmode: PatchPoolingMode = PatchPoolingMode.twostage,
        num_layers: int = 2,
        bidirectional: bool = False,
        **kwargs,
    ) -> None:

        if self.patchify:
            raise ValueError("Patches not yet supported for LSTM")
        super().__init__(
            feature_extractor,
            classifier,
            feature_size_in,
            feature_size_out,
            data_view_axis,
            patchify=patchify,
            num_patches=num_patches,
            patchmode=patchmode,
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
        patchify: bool,
        num_patches: int,
        patchmode: PatchPoolingMode = PatchPoolingMode.twostage,
        **transformer_kwargs,
    ) -> None:
        if self.patchify:
            raise ValueError("Patches not yet supported for Transformer")
        super().__init__(
            feature_extractor,
            classifier,
            feature_size_in,
            feature_size_out,
            data_view_axis,
            patchify=patchify,
            num_patches=num_patches,
            patchmode=patchmode,
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
