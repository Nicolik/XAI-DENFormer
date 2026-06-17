import torch
import torch.nn as nn

from classifier.model.utils import PositionalEncoding


# ----------------------------
# Encoder layer that can return attention maps
# ----------------------------
class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_weights=False):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False  # -> [B, nhead, L, L]
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if need_weights:
            return src, attn_weights
        return src


# ----------------------------
# Transformer Encoder Classifier (Backward compatible + chunking)
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

        # Keep a PE buffer (ckpt compatibility) but it will auto-extend per (sub)sequence length.
        pe_len = chunk_size if chunk_size is not None else max_len
        self.pos_enc = PositionalEncoding(d_model, max_len=pe_len)

        enc_layer = TransformerEncoderLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        # Preserve this module name for checkpoint compatibility
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # Will hold last attention maps returned by _encode when requested
        self.last_attn = None  # list[num_layers] -> list[num_chunks] -> Tensor[B, nhead, Lc, Lc]

    def forward(self, x, return_attn=False):
        if return_attn:
            pooled, attn = self._encode(x, return_attn=True)
            logits = self.head(pooled)
            return logits, attn
        else:
            pooled = self._encode(x, return_attn=False)
            logits = self.head(pooled)
            return logits

    def get_embedding(self, x):
        with torch.no_grad():
            pooled = self._encode(x, return_attn=False)
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

    def _encode(self, x, return_attn=False):
        """
        Returns:
            pooled: (B, D)
            attn: if requested, a nested list:
                  attn[layer_idx][chunk_idx] -> Tensor of shape [B, nhead, L_chunk, L_chunk]
                  For non-chunked mode, chunk_idx is always 0 (single chunk).
        """
        B, L, _ = x.shape
        x = self.input_proj(x)

        # Prepare attention collector
        num_layers = len(self.encoder.layers)
        attn_layers = [[] for _ in range(num_layers)] if return_attn else None

        if self.chunk_size is not None:
            chunks = x.split(self.chunk_size, dim=1)  # list of (B, Lc, D)
            chunk_reprs = []

            for chunk in chunks:
                h = self.pos_enc(chunk)  # PE auto-resizes to chunk length

                # Pass through each encoder layer, optionally collecting attention
                for li, layer in enumerate(self.encoder.layers):
                    if return_attn:
                        h, attn = layer(h, need_weights=True)
                        attn_layers[li].append(attn)  # [B, nhead, Lc, Lc]
                    else:
                        h = layer(h)

                chunk_repr = self._pool_tokens(h)  # (B, D)
                chunk_reprs.append(chunk_repr)

            # Pool across chunks. We keep mean aggregation here so the ablation only
            # changes token-to-chunk pooling: first vs mean vs max.
            chunk_reprs = torch.stack(chunk_reprs, dim=1)  # (B, num_chunks, D)
            pooled = chunk_reprs.mean(dim=1)               # (B, D)

        else:
            # Non-chunked path
            h = self.pos_enc(x)  # PE auto-resizes to full length
            for li, layer in enumerate(self.encoder.layers):
                if return_attn:
                    h, attn = layer(h, need_weights=True)
                    attn_layers[li].append(attn)  # single "chunk"
                else:
                    h = layer(h)
            pooled = self._pool_tokens(h)

        pooled = self.norm(pooled)

        if return_attn:
            self.last_attn = attn_layers
            return pooled, attn_layers
        return pooled
