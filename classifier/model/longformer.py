import math

import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel

from classifier.model.utils import PositionalEncoding, pool_tokens, validate_common_args


class LongformerClassifier(nn.Module):
    """
    Longformer classifier for already-embedded sequence inputs.

    Input:
        x: Tensor of shape (B, L, emb_dim)

    Notes:
        - Uses HuggingFace LongformerModel initialized from config, not pretrained text weights.
        - Uses inputs_embeds, so no tokenizer or input_ids are needed.
        - Uses local sliding-window attention only: global_attention_mask is all zeros.
        - max_position_embeddings includes a safety margin because HF Longformer pads
          sequences internally to a multiple of attention_window.
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
        attention_window=512,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()
        validate_common_args(d_model, nhead, pooling)

        if attention_window % 2 != 0:
            raise ValueError("Longformer attention_window must be even")

        self.pooling = pooling
        self.attention_window = int(attention_window)
        self.max_len = int(max_len)

        self.input_proj = nn.Linear(emb_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        # HuggingFace Longformer pads inputs internally to a multiple of attention_window.
        # Position IDs can therefore reach padded_len + 1. If max_len=12000 and
        # attention_window=512, padded_len=12288, so max_len + 2 is not enough.
        padded_max_len = int(math.ceil(max_len / attention_window) * attention_window)
        max_position_embeddings = padded_max_len + 2

        config = LongformerConfig(
            vocab_size=1,  # unused with inputs_embeds
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=nhead,
            intermediate_size=ff_dim,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_position_embeddings,
            attention_window=[attention_window] * num_layers,
            pad_token_id=0,
        )
        self.longformer = LongformerModel(config)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def _run_encoder(self, x):
        B, L, _ = x.shape
        if L > self.max_len:
            raise ValueError(
                f"Input sequence length L={L} exceeds configured max_len={self.max_len}. "
                f"Set MAX_LEN to at least the maximum sequence length in the dataset. "
                f"For this batch, use MAX_LEN >= {L}."
            )

        x = self.input_proj(x)
        x = self.pos_enc(x)

        attention_mask = torch.ones((B, L), dtype=torch.long, device=x.device)
        # All-zero global attention: controlled local-attention setting.
        global_attention_mask = torch.zeros((B, L), dtype=torch.long, device=x.device)

        out = self.longformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        return out.last_hidden_state

    def forward(self, x):
        h = self._run_encoder(x)
        pooled = self.norm(pool_tokens(h, self.pooling))
        return self.head(pooled)

    def get_embedding(self, x):
        with torch.no_grad():
            h = self._run_encoder(x)
            return self.norm(pool_tokens(h, self.pooling))
