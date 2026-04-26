"""
Training script for LoopMind full CNN+Transformer dual encoder.
Category-conditioned InfoNCE loss, all 4 categories trained jointly.

Run location : browser Colab (T4 GPU)
Checkpoints  : saved to Google Drive (ckpt_dir)
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

from data.extract_pairs import extract_pairs
from data.features       import extract_and_cache
from data.dataset        import SlakhPairDataset, CATEGORIES, CAT_TO_ID
from models.dual_encoder import DualEncoder
from losses.infonce      import infonce_loss


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True, help="Path to slakh2100 root")
    p.add_argument("--cache_dir",   default="data/cache")
    p.add_argument("--ckpt_dir",    default="checkpoints")
    p.add_argument("--max_tracks",  type=int,   default=400)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--hidden",      type=int,   default=256)
    p.add_argument("--embed_dim",   type=int,   default=128)
    p.add_argument("--n_heads",     type=int,   default=4)
    p.add_argument("--n_layers",    type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--resume",      action="store_true",
                   help="Resume from last.pt if it exists in ckpt_dir")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_batch_by_category(melody_batch, stem_batch, cat_ids):
    """
    Split a stacked batch into per-category sub-batches.
    Returns dict: {cat: (melody_tensor, stem_tensor)}
    """
    by_cat = {}
    for cat_idx, cat in enumerate(CATEGORIES):
        mask = (cat_ids == cat_idx)
        if mask.sum() < 2:
            continue
        by_cat[cat] = (melody_batch[mask], stem_batch[mask])
    return by_cat


def compute_recall(model, val_loader, device, k: int = 5):
    """R@K per category: fraction of queries where the GT pair is in top-K."""
    model.eval()
    all_qz = defaultdict(list)
    all_az = defaultdict(list)

    with torch.no_grad():
        for melody_batch, stem_batch, cat_ids in val_loader:
            by_cat = split_batch_by_category(
                melody_batch.to(device), stem_batch.to(device), cat_ids
            )
            for cat, (mf, sf) in by_cat.items():
                all_qz[cat].append(model.query_encoder(mf, cat).cpu())
                all_az[cat].append(model.audio_encoder(sf,  cat).cpu())

    recalls = {}
    for cat in CATEGORIES:
        if cat not in all_qz:
            continue
        Q   = torch.cat(all_qz[cat]).numpy()   # [N, D]
        A   = torch.cat(all_az[cat]).numpy()   # [N, D]
        sim = Q @ A.T                           # [N, N]
        # Number of items that score higher than the diagonal (true positive)
        ranks = (sim > np.diag(sim)[:, None]).sum(axis=1)
        recalls[cat] = float((ranks < k).mean())
    return recalls


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Data pipeline
    records  = extract_pairs(args.data_root, args.cache_dir,
                             max_tracks=args.max_tracks)
    features = extract_and_cache(records, args.cache_dir)

    train_ds = SlakhPairDataset(records, features, split="train")
    val_ds   = SlakhPairDataset(records, features, split="val")

    # Default collate_fn works now: all features are [T_SEQ, 128]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=2)

    # Model + optimiser
    model = DualEncoder(
        hidden=args.hidden, embed_dim=args.embed_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_mean_recall = 0.0
    train_losses     = []
    val_recalls_hist = []
    start_epoch      = 0

    # Resume from last checkpoint if requested and available
    last_ckpt = os.path.join(args.ckpt_dir, "last.pt")
    if args.resume and os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch      = ckpt["epoch"] + 1
        train_losses     = ckpt.get("train_losses", [])
        val_recalls_hist = ckpt.get("val_recalls_hist", [])
        best_mean_recall = ckpt.get("best_mean_recall", 0.0)
        # Advance scheduler to the correct step
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch} (best mean R@5 so far: {best_mean_recall:.3f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        cat_losses = defaultdict(float)
        cat_counts = defaultdict(int)
        n_batches  = 0

        for melody_batch, stem_batch, cat_ids in train_loader:
            melody_batch = melody_batch.to(device)
            stem_batch   = stem_batch.to(device)

            by_cat = split_batch_by_category(melody_batch, stem_batch, cat_ids)
            if not by_cat:
                continue

            total_loss = torch.tensor(0.0, device=device)
            n_active   = 0
            for cat, (mf, sf) in by_cat.items():
                zq   = model.query_encoder(mf, cat)
                za   = model.audio_encoder(sf,  cat)
                loss = infonce_loss(zq, za, args.temperature)
                total_loss += loss
                cat_losses[cat] += loss.item()
                cat_counts[cat] += 1
                n_active += 1

            total_loss = total_loss / n_active
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Per-category loss string for quick debugging
        cat_loss_str = " | ".join(
            f"{c}={cat_losses[c]/max(cat_counts[c],1):.4f}" for c in CATEGORIES
        )

        recalls     = compute_recall(model, val_loader, device, k=5)
        mean_recall = float(np.mean(list(recalls.values()))) if recalls else 0.0
        val_recalls_hist.append(recalls)

        recall_str = " | ".join(f"{c}={recalls.get(c, 0):.3f}" for c in CATEGORIES)
        print(f"Epoch {epoch+1:03d} | loss={avg_loss:.4f} [{cat_loss_str}] "
              f"| R@5: {recall_str} | mean={mean_recall:.3f}")

        ckpt = {
            "epoch":             epoch,
            "model":             model.state_dict(),
            "optimizer":         optimizer.state_dict(),
            "train_losses":      train_losses,
            "val_recalls_hist":  val_recalls_hist,
            "best_mean_recall":  best_mean_recall,
            "args":              vars(args),
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, "last.pt"))

        if mean_recall > best_mean_recall:
            best_mean_recall = mean_recall
            ckpt["best_mean_recall"] = best_mean_recall
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pt"))
            print(f"  → New best: mean R@5 = {best_mean_recall:.3f}")

    print(f"\nTraining done. Best mean R@5: {best_mean_recall:.3f}")


if __name__ == "__main__":
    train()
