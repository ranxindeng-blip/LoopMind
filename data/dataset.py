"""
PyTorch Dataset for demo training.
Returns (melody_feat, stem_feat, category_id) triples.

Run location : local Mac
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset

CATEGORIES   = ["drums", "bass", "piano", "guitar"]
CAT_TO_ID    = {c: i for i, c in enumerate(CATEGORIES)}


class SlakhPairDataset(Dataset):
    """
    Each item: (melody_feat, stem_feat, category_id)
    Positive pair  = melody from track A + stem from track A (same category)
    Negatives      = all other stems in the batch (implicit in InfoNCE)
    """

    def __init__(self, records: list, features: dict,
                 split: str = "train", val_ratio: float = 0.2):
        self.pairs = []

        # Split tracks into train / val
        tracks = [r["track"] for r in records]
        random.seed(42)
        random.shuffle(tracks)
        n_val  = max(1, int(len(tracks) * val_ratio))
        val_tracks = set(tracks[:n_val])

        for rec in records:
            track = rec["track"]
            if split == "train" and track in val_tracks:
                continue
            if split == "val" and track not in val_tracks:
                continue

            melody_feat = features[track]["melody"]
            if melody_feat is None:
                continue

            for cat in CATEGORIES:
                for audio_path, stem_feat in features[track][cat].items():
                    self.pairs.append((
                        melody_feat.copy(),
                        stem_feat.copy(),
                        CAT_TO_ID[cat],
                    ))

        print(f"[{split}] {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mel_feat, stem_feat, cat_id = self.pairs[idx]
        return (
            torch.from_numpy(mel_feat),
            torch.from_numpy(stem_feat),
            cat_id,
        )
