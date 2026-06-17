import math
import torch
import torch.nn as nn


def pool_tokens(h, pooling):
    if pooling == "first":
        return h[:, 0, :]
    if pooling == "mean":
        return h.mean(dim=1)
    if pooling == "max":
        return h.max(dim=1).values
    raise ValueError(f"Unsupported pooling={pooling!r}. Choose one of: first, mean, max")


def validate_common_args(d_model, nhead, pooling):
    if d_model % nhead != 0:
        raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
    if pooling not in {"first", "mean", "max"}:
        raise ValueError(f"Unsupported pooling={pooling!r}. Choose one of: first, mean, max")


# ----------------------------
# Positional Encoding (buffer kept for ckpt compat, auto-extends on demand)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        pe = self._build_pe(max_len, d_model, device="cpu")
        self.register_buffer("pe", pe)  # (1, max_len, d_model)

    @staticmethod
    def _build_pe(L, D, device):
        pe = torch.zeros(L, D, device=device)
        position = torch.arange(0, L, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float, device=device) *
                             (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, L, D)

    def _ensure_len(self, L, device):
        # Extend or move buffer if needed (keeps backward compatibility with old checkpoints)
        if self.pe.device != device or self.pe.size(1) < L:
            with torch.no_grad():
                new_pe = self._build_pe(L, self.d_model, device=device)
                self.pe = new_pe  # overwrite buffer with new size/device

    def forward(self, x):  # x: (B, L, D)
        B, L, D = x.shape
        self._ensure_len(L, x.device)
        return x + self.pe[:, :L, :]
