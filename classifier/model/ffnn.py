import torch
import torch.nn as nn


class FFNNClassifier(nn.Module):
    """
    Flatten-based feed-forward neural network classifier.

    Input:
        x: Tensor of shape (B, L, emb_dim)

    The full padded sequence is flattened to preserve positional information,
    then classified with a standard MLP. This makes it comparable to the
    flatten-based LogisticRegressionClassifier.
    """

    def __init__(
        self,
        emb_dim,
        d_model,
        nhead,
        ff_dim,
        num_layers,
        num_classes,
        max_len=12000,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.max_len = max_len
        input_dim = max_len * emb_dim

        layers = []
        in_dim = input_dim

        for _ in range(max(1, num_layers)):
            layers.extend([
                nn.Linear(in_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = ff_dim

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, num_classes)

    def _encode(self, x):
        B, L, D = x.shape

        if D != self.emb_dim:
            raise ValueError(f"Expected emb_dim={self.emb_dim}, got D={D}")

        if L > self.max_len:
            raise ValueError(
                f"Input sequence length L={L} exceeds max_len={self.max_len}. "
                "Pad/truncate consistently before calling FFNNClassifier."
            )

        if L < self.max_len:
            pad_len = self.max_len - L
            pad = x.new_zeros(B, pad_len, D)
            x = torch.cat([x, pad], dim=1)

        x = x.reshape(B, self.max_len * self.emb_dim)
        return self.encoder(x)

    def forward(self, x):
        features = self._encode(x)
        return self.head(features)

    def get_embedding(self, x):
        with torch.no_grad():
            return self._encode(x)