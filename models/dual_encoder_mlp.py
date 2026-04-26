"""
Legacy MLP dual encoder — used by demo/app.py with the BabySlakh checkpoint.
Do not modify; the saved checkpoint (best.pt) was trained with this architecture.
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
    def __init__(self, in_dim: int = 24, hidden: int = 128, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = _mlp(in_dim, hidden, hidden)
        self.heads = nn.ModuleDict({
            cat: nn.Linear(hidden, embed_dim) for cat in CATEGORIES
        })

    def forward(self, x: torch.Tensor, category: str) -> torch.Tensor:
        h = F.relu(self.backbone(x))
        return F.normalize(self.heads[category](h), dim=-1)

    def forward_all(self, x: torch.Tensor) -> dict:
        h = F.relu(self.backbone(x))
        return {cat: F.normalize(self.heads[cat](h), dim=-1) for cat in CATEGORIES}


class AudioEncoder(nn.Module):
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
        return F.normalize(self.heads[category](h), dim=-1)


class DualEncoder(nn.Module):
    def __init__(self, hidden: int = 128, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.query_encoder = QueryEncoder(hidden=hidden, embed_dim=embed_dim)
        self.audio_encoder = AudioEncoder(hidden=hidden, embed_dim=embed_dim)

    def forward(self, melody_feat, stem_feat, category):
        return self.query_encoder(melody_feat, category), \
               self.audio_encoder(stem_feat, category)

    def encode_query(self, melody_feat: torch.Tensor) -> dict:
        return self.query_encoder.forward_all(melody_feat)

    def encode_audio(self, stem_feat: torch.Tensor, category: str) -> torch.Tensor:
        return self.audio_encoder(stem_feat, category)
