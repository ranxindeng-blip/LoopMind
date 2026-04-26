"""
Training script — demo version.
Trains DualEncoder with category-conditioned InfoNCE in one joint pass.

Run location : local Mac (no GPU needed for demo MLP)
Checkpoints  : saved locally to checkpoints/best.pt
"""
import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np

from data.extract_pairs import extract_pairs
from data.features       import extract_and_cache
from data.dataset        import SlakhPairDataset, CATEGORIES, CAT_TO_ID
from models.dual_encoder import DualEncoder
from losses.infonce      import infonce_loss


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  required=True, help="Path to babyslakh_16k root")
    p.add_argument("--cache_dir",  default="data/cache")
    p.add_argument("--ckpt_dir",   default="checkpoints")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--temperature",type=float, default=0.07)
    p.add_argument("--hidden",     type=int,   default=128)
    p.add_argument("--embed_dim",  type=int,   default=64)
    return p.parse_args()


def collate_by_category(batch):
    """Group batch items by category for joint loss computation."""
    by_cat = defaultdict(lambda: {"melody": [], "stem": []})
    for melody_feat, stem_feat, cat_id in batch:
        cat = CATEGORIES[cat_id]
        by_cat[cat]["melody"].append(melody_feat)
        by_cat[cat]["stem"].append(stem_feat)
    return by_cat


def compute_recall(model, val_loader, device, k=5):
    """R@K: fraction of queries where the true positive is in top-K."""
    model.eval()
    all_qz, all_az, all_cats = defaultdict(list), defaultdict(list), defaultdict(list)

    with torch.no_grad():
        for batch in val_loader:
            by_cat = collate_by_category(batch)
            for cat, tensors in by_cat.items():
                if len(tensors["melody"]) < 2:
                    continue
                mf = torch.stack(tensors["melody"]).to(device)
                sf = torch.stack(tensors["stem"]).to(device)
                zq = model.query_encoder(mf, cat)
                za = model.audio_encoder(sf, cat)
                all_qz[cat].append(zq.cpu())
                all_az[cat].append(za.cpu())

    recalls = {}
    for cat in CATEGORIES:
        if cat not in all_qz:
            continue
        Q = torch.cat(all_qz[cat])   # [N, D]
        A = torch.cat(all_az[cat])   # [N, D]
        sim = (Q @ A.T).numpy()      # [N, N]
        ranks = (sim > np.diag(sim)[:, None]).sum(axis=1)
        recalls[cat] = float((ranks < k).mean())

    return recalls


def train():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Data
    records  = extract_pairs(args.data_root, args.cache_dir)
    features = extract_and_cache(records, args.cache_dir)

    train_ds = SlakhPairDataset(records, features, split="train")
    val_ds   = SlakhPairDataset(records, features, split="val")

    # collate_fn=list: disable auto-stacking since drums(256-d) != others(24-d)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=False,
                              collate_fn=list)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False,
                              collate_fn=list)

    # Model
    model     = DualEncoder(hidden=args.hidden, embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mean_recall = 0.0
    train_losses, val_recalls_history = [], []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            by_cat = collate_by_category(batch)
            z_queries, z_audios = {}, {}

            for cat, tensors in by_cat.items():
                if len(tensors["melody"]) < 2:
                    continue
                mf = torch.stack(tensors["melody"]).to(device)
                sf = torch.stack(tensors["stem"]).to(device)
                z_queries[cat] = model.query_encoder(mf, cat)
                z_audios[cat]  = model.audio_encoder(sf, cat)

            if not z_queries:
                continue

            loss = torch.tensor(0.0, device=device)
            n_cats = 0
            for cat in z_queries:
                if z_queries[cat].size(0) >= 2:
                    loss += infonce_loss(z_queries[cat], z_audios[cat], args.temperature)
                    n_cats += 1
            loss = loss / max(n_cats, 1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        recalls      = compute_recall(model, val_loader, device, k=5)
        mean_recall  = np.mean(list(recalls.values())) if recalls else 0.0
        val_recalls_history.append(recalls)

        recall_str = " | ".join(f"{c}={v:.3f}" for c, v in recalls.items())
        print(f"Epoch {epoch+1:03d} | loss={avg_loss:.4f} | R@5: {recall_str} | mean={mean_recall:.3f}")

        ckpt = {
            "epoch":              epoch,
            "model":              model.state_dict(),
            "optimizer":          optimizer.state_dict(),
            "train_losses":       train_losses,
            "val_recalls_history": val_recalls_history,
            "best_mean_recall":   best_mean_recall,
            "args":               vars(args),
        }
        # Always save last checkpoint
        torch.save(ckpt, os.path.join(args.ckpt_dir, "last.pt"))

        if mean_recall > best_mean_recall:
            best_mean_recall = mean_recall
            ckpt["best_mean_recall"] = best_mean_recall
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pt"))
            print(f"  → Saved best (mean R@5={best_mean_recall:.3f})")

    print(f"\nTraining done. Best mean R@5: {best_mean_recall:.3f}")


if __name__ == "__main__":
    train()
