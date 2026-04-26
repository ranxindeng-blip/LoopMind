"""
Dual encoder for demo version.

Shared MLP backbone + 4 category-specific projection heads per side.

Query  side : melody chroma [24-d]  → shared MLP → head → 64-d
Audio  side : drums mel [256-d]     → shared MLP → head → 64-d
           : bass/piano/guitar chroma [24-d] → shared MLP → head → 64-d

Run location : local Mac
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CATEGORIES = ["drums", "bass", "piano", "guitar"]
EMBED_DIM  = 64


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class QueryEncoder(nn.Module):
    """
    Melody chroma [24-d] → shared backbone → 4 projection heads → 64-d per category.
    Input dim is always 24 (chroma mean+std) for all categories.
    """

    def __init__(self, in_dim: int = 24, hidden: int = 128, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = _mlp(in_dim, hidden, hidden)
        self.heads = nn.ModuleDict({
            cat: nn.Linear(hidden, embed_dim) for cat in CATEGORIES
        })

    def forward(self, x: torch.Tensor, category: str) -> torch.Tensor:
        h = F.relu(self.backbone(x))
        z = self.heads[category](h)
        return F.normalize(z, dim=-1)

    def forward_all(self, x: torch.Tensor) -> dict:
        """Return embeddings for all 4 categories at once (used at inference)."""
        h = F.relu(self.backbone(x))
        return {cat: F.normalize(self.heads[cat](h), dim=-1) for cat in CATEGORIES}


class AudioEncoder(nn.Module):
    """
    Per-category audio feature → shared backbone → projection head → 64-d.
    drums  : in_dim = 256 (mel mean+std)
    others : in_dim = 24  (chroma mean+std)
    Each category has its own backbone + head (different input dims).
    """

    def __init__(self, hidden: int = 128, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoders = nn.ModuleDict({
            "drums":  _mlp(256, hidden, hidden),
            "bass":   _mlp(24,  hidden, hidden),
            "piano":  _mlp(24,  hidden, hidden),
            "guitar": _mlp(24,  hidden, hidden),
        })
        self.heads = nn.ModuleDict({
            cat: nn.Linear(hidden, embed_dim) for cat in CATEGORIES
        })

    def forward(self, x: torch.Tensor, category: str) -> torch.Tensor:
        h = F.relu(self.encoders[category](x))
        z = self.heads[category](h)
        return F.normalize(z, dim=-1)


class DualEncoder(nn.Module):
    def __init__(self, hidden: int = 128, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.query_encoder = QueryEncoder(hidden=hidden, embed_dim=embed_dim)
        self.audio_encoder = AudioEncoder(hidden=hidden, embed_dim=embed_dim)

    def forward(self, melody_feat: torch.Tensor,
                stem_feat: torch.Tensor,
                category: str):
        z_query = self.query_encoder(melody_feat, category)
        z_audio = self.audio_encoder(stem_feat, category)
        return z_query, z_audio

    def encode_query(self, melody_feat: torch.Tensor) -> dict:
        return self.query_encoder.forward_all(melody_feat)

    def encode_audio(self, stem_feat: torch.Tensor, category: str) -> torch.Tensor:
        return self.audio_encoder(stem_feat, category)
