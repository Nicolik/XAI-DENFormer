import torch
import torch.nn as nn

from performer_pytorch import Performer

from classifier.model.utils import PositionalEncoding, pool_tokens, validate_common_args


class PerformerClassifier(nn.Module):
    """
    Performer classifier for already-embedded sequence inputs.

    Input:
        x: Tensor of shape (B, L, emb_dim)
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
        nb_features=256,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()
        validate_common_args(d_model, nhead, pooling)

        self.pooling = pooling
        self.input_proj = nn.Linear(emb_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        ff_mult = max(1, ff_dim // d_model)
        self.performer = Performer(
            dim=d_model,
            depth=num_layers,
            heads=nhead,
            dim_head=d_model // nhead,
            causal=False,
            nb_features=nb_features,
            ff_mult=ff_mult,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        h = self.performer(x)
        pooled = self.norm(pool_tokens(h, self.pooling))
        return self.head(pooled)

    def get_embedding(self, x):
        with torch.no_grad():
            x = self.input_proj(x)
            x = self.pos_enc(x)
            h = self.performer(x)
            return self.norm(pool_tokens(h, self.pooling))
