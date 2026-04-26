"""
PyTorch Dataset for LoopMind full model.

Each item: (melody_feat [T,128], stem_feat [T,128], category_id)
All features share the same shape, so the default DataLoader collate_fn works.

Slakh2100 split convention:
  - records with split="train"      → training set
  - records with split="validation" → validation set
If all records share the same split value (e.g. only "train" subset was loaded),
a random 80/20 split is applied instead.

Run location : browser Colab
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset

CATEGORIES = ["drums", "bass", "piano", "guitar"]
CAT_TO_ID  = {c: i for i, c in enumerate(CATEGORIES)}


class SlakhPairDataset(Dataset):
    """
    Positive pair = melody from track X + one accompaniment stem from track X.
    Negatives are all other items in the batch (implicit in InfoNCE loss).
    """

    def __init__(self, records: list, features: dict, split: str = "train"):
        self.pairs = self._build_pairs(records, features, split)
        print(f"[{split}] {len(self.pairs)} pairs across "
              f"{len(set(p[3] for p in self.pairs))} tracks")

    @staticmethod
    def _build_pairs(records: list, features: dict, split: str) -> list:
        # Determine whether to use pre-defined splits or fall back to random
        unique_splits = {r.get("split", "train") for r in records}
        use_predefined = len(unique_splits) > 1

        if use_predefined:
            target_split = "validation" if split == "val" else split
            split_tracks = {r["track"] for r in records
                            if r.get("split", "train") == target_split}
        else:
            # All records have the same split label → random 80/20
            tracks = [r["track"] for r in records]
            random.seed(42)
            random.shuffle(tracks)
            n_val = max(1, int(len(tracks) * 0.2))
            val_set = set(tracks[:n_val])
            split_tracks = val_set if split == "val" else set(tracks[n_val:])

        pairs = []
        for rec in records:
            track = rec["track"]
            if track not in split_tracks:
                continue
            melody_feat = features.get(track, {}).get("melody")
            if melody_feat is None:
                continue
            for cat in CATEGORIES:
                for audio_path, stem_feat in features[track].get(cat, {}).items():
                    pairs.append((
                        melody_feat.copy(),
                        stem_feat.copy(),
                        CAT_TO_ID[cat],
                        track,          # kept for diagnostics; not used by model
                    ))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        mel_feat, stem_feat, cat_id, _ = self.pairs[idx]
        return (
            torch.from_numpy(mel_feat),   # [T, 128]
            torch.from_numpy(stem_feat),  # [T, 128]
            cat_id,
        )
