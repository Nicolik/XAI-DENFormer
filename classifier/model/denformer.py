import torch
import torch.nn as nn

from classifier.model.utils import PositionalEncoding


# ----------------------------
# Transformer Encoder Classifier
# ----------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, emb_dim, d_model, nhead, ff_dim, num_layers, num_classes,
                 max_len=12000, chunk_size=None, dropout=0.1, pooling="first"):
        super().__init__()
        if pooling not in {"first", "mean", "max"}:
            raise ValueError(f"Unsupported pooling='{pooling}'. Choose one of: first, mean, max.")

        self.chunk_size = chunk_size
        self.pooling = pooling
        self.input_proj = nn.Linear(emb_dim, d_model)

        pe_len = chunk_size if chunk_size is not None else max_len
        self.pos_enc = PositionalEncoding(d_model, max_len=pe_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        pooled = self._encode(x)
        logits = self.head(pooled)
        return logits

    def get_embedding(self, x):
        """Return the pooled embedding (before classification head)."""
        with torch.no_grad():  # disable gradients for inference
            pooled = self._encode(x)
        return pooled

    def _pool_tokens(self, h):
        """
        Pool token representations into one sequence/chunk representation.

        Args:
            h: Tensor of shape (B, L, D)

        Returns:
            Tensor of shape (B, D)
        """
        if self.pooling == "first":
            return h[:, 0, :]
        if self.pooling == "mean":
            return h.mean(dim=1)
        if self.pooling == "max":
            return h.max(dim=1).values
        raise ValueError(f"Unsupported pooling='{self.pooling}'. Choose one of: first, mean, max.")

    def _encode(self, x):
        """Shared sequence encoding logic (used in forward + embedding)."""
        B, L, _ = x.shape
        x = self.input_proj(x)

        if self.chunk_size is not None:
            chunks = x.split(self.chunk_size, dim=1)
            chunk_reprs = []
            for chunk in chunks:
                h = self.pos_enc(chunk)
                h = self.encoder(h)
                chunk_repr = self._pool_tokens(h)  # (B, D)
                chunk_reprs.append(chunk_repr)

            # Pool across chunks. We keep mean aggregation here so the ablation only
            # changes token-to-chunk pooling: first vs mean vs max.
            chunk_reprs = torch.stack(chunk_reprs, dim=1)  # (B, num_chunks, D)
            pooled = chunk_reprs.mean(dim=1)               # (B, D)
        else:
            h = self.pos_enc(x)
            h = self.encoder(h)
            pooled = self._pool_tokens(h)

        pooled = self.norm(pooled)
        return pooled
