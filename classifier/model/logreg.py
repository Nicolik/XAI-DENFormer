import torch
import torch.nn as nn


class LogisticRegressionClassifier(nn.Module):
    """
    Multinomial logistic-regression classifier with the same workflow API as the
    neural sequence models.

    Input:
        x: Tensor of shape (B, L, emb_dim)

    The sequence is flattened into a fixed-size vector, then a single linear
    layer maps it to class logits. This is equivalent to multinomial logistic
    regression on the full padded OHE sequence.
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
        dropout=0.0,
        pooling="mean",
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(max_len * emb_dim, num_classes)

    def _encode(self, x):
        B, L, D = x.shape

        if D != self.emb_dim:
            raise ValueError(f"Expected emb_dim={self.emb_dim}, got D={D}")

        if L > self.max_len:
            raise ValueError(
                f"Input sequence length L={L} exceeds max_len={self.max_len}. "
                "Pad/truncate consistently before calling LogisticRegressionClassifier."
            )

        if L < self.max_len:
            pad_len = self.max_len - L
            pad = x.new_zeros(B, pad_len, D)
            x = torch.cat([x, pad], dim=1)

        x = x.reshape(B, self.max_len * self.emb_dim)
        return self.dropout(x)

    def forward(self, x):
        features = self._encode(x)
        return self.head(features)

    def get_embedding(self, x):
        with torch.no_grad():
            return self._encode(x)