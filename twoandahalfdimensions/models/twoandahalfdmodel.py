from torch import Tensor, nn


class TwoAndAHalfDModel(nn.Module):
    def __init__(
        self, feature_extractor: nn.Module, classifier: nn.Module, hidden_size: int
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.att = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.classifier = classifier

    def forward(self, x: Tensor):
        N_scans, C, N_slices, px_x, px_y = x.shape
        x.transpose_(1, 2)
        x = x.reshape(N_scans * N_slices, C, px_x, px_y)
        features = self.feature_extractor(x)  # shape (N_scans*N_slices, num_classes)
        features = features.reshape(
            N_scans, N_slices, -1
        )  # shape (N_scans, N_slices, num_classes)

        query = features.mean(1, keepdims=True)
        features, att_map = self.att(query, features, features)
        features.squeeze_(1)
        att_map.squeeze_(1)
        out = self.classifier(features)
        return out, att_map
