"""
Evaluation: R@1, R@5, R@10 per category + loss curves.

Run location : browser Colab
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader

from data.extract_pairs import extract_pairs
from data.features       import extract_and_cache
from data.dataset        import SlakhPairDataset, CATEGORIES
from models.dual_encoder import DualEncoder


def evaluate(ckpt_path: str, data_root: str, cache_dir: str,
             max_tracks: int = 400, plot_path: str = "loss_curves.png"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    args   = ckpt["args"]

    model = DualEncoder(
        hidden    = args["hidden"],
        embed_dim = args["embed_dim"],
        n_heads   = args.get("n_heads",  4),
        n_layers  = args.get("n_layers", 2),
        dropout   = args.get("dropout",  0.1),
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    records  = extract_pairs(data_root, cache_dir, max_tracks=max_tracks)
    features = extract_and_cache(records, cache_dir)
    val_ds   = SlakhPairDataset(records, features, split="val")
    loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    all_qz = defaultdict(list)
    all_az = defaultdict(list)

    with torch.no_grad():
        for melody_batch, stem_batch, cat_ids in loader:
            melody_batch = melody_batch.to(device)
            stem_batch   = stem_batch.to(device)
            for cat_idx, cat in enumerate(CATEGORIES):
                mask = (cat_ids == cat_idx)
                if mask.sum() < 1:
                    continue
                mf = melody_batch[mask]
                sf = stem_batch[mask]
                all_qz[cat].append(model.query_encoder(mf, cat).cpu())
                all_az[cat].append(model.audio_encoder(sf,  cat).cpu())

    print("\n=== Evaluation Results ===\n")
    print(f"{'Category':<12} {'R@1':>6} {'R@5':>6} {'R@10':>6}  {'Random':>8}")

    for cat in CATEGORIES:
        if cat not in all_qz:
            print(f"{cat:<12}  — not enough data")
            continue
        Q   = torch.cat(all_qz[cat]).numpy()
        A   = torch.cat(all_az[cat]).numpy()
        N   = Q.shape[0]
        sim = Q @ A.T
        ranks = (sim > np.diag(sim)[:, None]).sum(axis=1)
        r1  = float((ranks < 1).mean())
        r5  = float((ranks < 5).mean())
        r10 = float((ranks < 10).mean())
        print(f"{cat:<12} {r1:>6.3f} {r5:>6.3f} {r10:>6.3f}  {1/N:>8.3f}")

    # Loss + recall curves
    train_losses = ckpt.get("train_losses", [])
    val_recalls  = ckpt.get("val_recalls_hist", [])
    if train_losses:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(train_losses, color="steelblue", linewidth=2)
        axes[0].set(title="Training Loss (InfoNCE)", xlabel="Epoch", ylabel="Loss")
        axes[0].grid(True, alpha=0.3)
        if val_recalls:
            for cat in CATEGORIES:
                vals = [r.get(cat, 0) for r in val_recalls]
                axes[1].plot(vals, label=cat, linewidth=1.5)
            axes[1].set(title="Val R@5 per Category", xlabel="Epoch", ylabel="R@5")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path",  default="checkpoints/best.pt")
    p.add_argument("--data_root",  required=True)
    p.add_argument("--cache_dir",  default="data/cache")
    p.add_argument("--max_tracks", type=int, default=400)
    args = p.parse_args()
    evaluate(args.ckpt_path, args.data_root, args.cache_dir, args.max_tracks)
