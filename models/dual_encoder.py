"""
Full CNN + Transformer dual encoder for LoopMind.

Both modalities share the same input shape: [B, T_SEQ, 128]
  - Query  : piano roll sequence   (MIDI melody)
  - Audio  : log-mel spectrogram   (accompaniment stem)

Architecture per encoder:
  Conv1D stack (local patterns) → Transformer (global context) → GAP → projection head

QueryEncoder  : one shared Conv+Transformer backbone, 4 category-specific linear heads
AudioEncoder  : 4 independent Conv+Transformer backbones (one per category), one head each

Run location : browser Colab
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CATEGORIES = ["drums", "bass", "piano", "guitar"]
IN_CHANNELS = 128   # piano roll bins / mel bins
HIDDEN      = 256   # internal feature width
EMBED_DIM   = 128   # output embedding dimension


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two Conv1D layers with BatchNorm + GELU activation. Input/output: [B, C, T]."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        # Residual projection if channel dims differ
        self.residual = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class Backbone(nn.Module):
    """
    Shared encoder backbone.
    Input  : [B, T, in_channels]
    Output : [B, hidden]   (global average pool over time)
    """

    def __init__(self, in_channels: int = IN_CHANNELS, hidden: int = HIDDEN,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv = ConvBlock(in_channels, hidden, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,    # [B, T, hidden]
            norm_first=True,     # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_channels]
        x = x.permute(0, 2, 1)          # [B, in_channels, T]
        x = self.conv(x)                 # [B, hidden, T]
        x = x.permute(0, 2, 1)          # [B, T, hidden]
        x = self.transformer(x)          # [B, T, hidden]
        x = x.mean(dim=1)               # [B, hidden]  — global average pool
        return x


# ── Encoders ──────────────────────────────────────────────────────────────────

class QueryEncoder(nn.Module):
    """
    MIDI piano roll → embedding per category.
    One shared Backbone + 4 linear projection heads.
    """

    def __init__(self, hidden: int = HIDDEN, embed_dim: int = EMBED_DIM,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = Backbone(IN_CHANNELS, hidden, n_heads, n_layers, dropout)
        self.heads = nn.ModuleDict({
            cat: nn.Linear(hidden, embed_dim) for cat in CATEGORIES
        })

    def forward(self, x: torch.Tensor, category: str) -> torch.Tensor:
        """x: [B, T, 128] → [B, embed_dim] L2-normalised."""
        h = self.backbone(x)
        return F.normalize(self.heads[category](h), dim=-1)

    def forward_all(self, x: torch.Tensor) -> dict:
        """Return embeddings for all 4 categories (used at demo inference)."""
        h = self.backbone(x)
        return {cat: F.normalize(self.heads[cat](h), dim=-1) for cat in CATEGORIES}


class AudioEncoder(nn.Module):
    """
    Audio mel spectrogram → embedding for one category.
    4 independent Backbone+head pairs — each category has different audio patterns.
    """

    def __init__(self, hidden: int = HIDDEN, embed_dim: int = EMBED_DIM,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbones = nn.ModuleDict({
            cat: Backbone(IN_CHANNELS, hidden, n_heads, n_layers, dropout)
            for cat in CATEGORIES
        })
        self.heads = nn.ModuleDict({
            cat: nn.Linear(hidden, embed_dim) for cat in CATEGORIES
        })

    def forward(self, x: torch.Tensor, category: str) -> torch.Tensor:
        """x: [B, T, 128] → [B, embed_dim] L2-normalised."""
        h = self.backbones[category](x)
        return F.normalize(self.heads[category](h), dim=-1)


# ── Unified DualEncoder ───────────────────────────────────────────────────────

class DualEncoder(nn.Module):
    def __init__(self, hidden: int = HIDDEN, embed_dim: int = EMBED_DIM,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.query_encoder = QueryEncoder(hidden, embed_dim, n_heads, n_layers, dropout)
        self.audio_encoder = AudioEncoder(hidden, embed_dim, n_heads, n_layers, dropout)

    def forward(self, melody_feat: torch.Tensor,
                stem_feat: torch.Tensor,
                category: str):
        """Return (query_emb, audio_emb) for one category."""
        z_q = self.query_encoder(melody_feat, category)
        z_a = self.audio_encoder(stem_feat,   category)
        return z_q, z_a

    def encode_query(self, melody_feat: torch.Tensor) -> dict:
        """All-category query embeddings for demo inference."""
        return self.query_encoder.forward_all(melody_feat)

    def encode_audio(self, stem_feat: torch.Tensor, category: str) -> torch.Tensor:
        return self.audio_encoder(stem_feat, category)
