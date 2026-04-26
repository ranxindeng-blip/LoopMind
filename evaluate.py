"""
Evaluation: R@1, R@5, R@10 per category + ablation table + loss curves.

Run location : local Mac
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader

from data.extract_pairs import extract_pairs
from data.features       import extract_and_cache
from data.dataset        import SlakhPairDataset, CATEGORIES
from models.dual_encoder import DualEncoder


def evaluate(ckpt_path: str, data_root: str, cache_dir: str,
             plot_path: str = "loss_curves.png"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    args   = ckpt["args"]

    model  = DualEncoder(hidden=args["hidden"], embed_dim=args["embed_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    records  = extract_pairs(data_root, cache_dir)
    features = extract_and_cache(records, cache_dir)
    test_ds  = SlakhPairDataset(records, features, split="val")
    loader   = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_qz = defaultdict(list)
    all_az = defaultdict(list)

    from data.dataset import CATEGORIES as CATS
    with torch.no_grad():
        for batch in loader:
            from train import collate_by_category
            by_cat = collate_by_category(batch)
            for cat, tensors in by_cat.items():
                if len(tensors["melody"]) < 1:
                    continue
                mf = torch.stack(tensors["melody"]).to(device)
                sf = torch.stack(tensors["stem"]).to(device)
                all_qz[cat].append(model.query_encoder(mf, cat).cpu())
                all_az[cat].append(model.audio_encoder(sf, cat).cpu())

    print("\n=== Evaluation Results ===\n")
    print(f"{'Category':<12} {'R@1':>6} {'R@5':>6} {'R@10':>6}  {'Random R@1':>10}")

    for cat in CATEGORIES:
        if cat not in all_qz:
            print(f"{cat:<12} — not enough data")
            continue
        Q = torch.cat(all_qz[cat]).numpy()
        A = torch.cat(all_az[cat]).numpy()
        N = Q.shape[0]
        sim = Q @ A.T   # [N, N]
        ranks = (sim > np.diag(sim)[:, None]).sum(axis=1)

        r1   = float((ranks < 1).mean())
        r5   = float((ranks < 5).mean())
        r10  = float((ranks < 10).mean())
        rand = 1.0 / N
        print(f"{cat:<12} {r1:>6.3f} {r5:>6.3f} {r10:>6.3f}  {rand:>10.3f}")

    # Loss curves
    train_losses = ckpt.get("train_losses", [])
    if train_losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, color="steelblue", linewidth=2)
        ax.set(title="Training Loss (InfoNCE)", xlabel="Epoch", ylabel="Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"\nLoss curve saved to {plot_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path",  default="checkpoints/best.pt")
    p.add_argument("--data_root",  required=True)
    p.add_argument("--cache_dir",  default="data/cache")
    args = p.parse_args()
    evaluate(args.ckpt_path, args.data_root, args.cache_dir)
