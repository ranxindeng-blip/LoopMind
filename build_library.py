"""
Pre-compute audio embeddings for all stems → stem library.
Run once after training. Library is used at demo inference time.

Run location : local Mac
Output stored : locally to data/cache/library.pt
"""
import os
import torch
import numpy as np
from collections import defaultdict

from data.extract_pairs import extract_pairs
from data.features       import extract_and_cache
from models.dual_encoder import DualEncoder, CATEGORIES


def build_library(data_root: str, cache_dir: str, ckpt_path: str,
                  library_path: str = "data/cache/library.pt"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt  = torch.load(ckpt_path, map_location=device)
    args  = ckpt["args"]
    model = DualEncoder(hidden=args["hidden"], embed_dim=args["embed_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    records  = extract_pairs(data_root, cache_dir)
    features = extract_and_cache(records, cache_dir)

    library = {cat: {"embeddings": [], "paths": [], "tracks": []} for cat in CATEGORIES}

    with torch.no_grad():
        for rec in records:
            track = rec["track"]
            for cat in CATEGORIES:
                for audio_path, feat in features[track][cat].items():
                    x = torch.from_numpy(feat).unsqueeze(0).to(device)
                    z = model.encode_audio(x, cat).squeeze(0).cpu().numpy()
                    library[cat]["embeddings"].append(z)
                    library[cat]["paths"].append(audio_path)
                    library[cat]["tracks"].append(track)

    # Stack embeddings
    for cat in CATEGORIES:
        library[cat]["embeddings"] = np.stack(library[cat]["embeddings"])
        print(f"  {cat}: {len(library[cat]['paths'])} stems")

    torch.save(library, library_path)
    print(f"\nLibrary saved to {library_path}")
    return library


def retrieve(query_emb: np.ndarray, library: dict, category: str,
             top_k: int = 3, exclude_track: str = None) -> list:
    """
    Cosine similarity search within one category.
    Returns list of (audio_path, score, track).
    """
    embs   = library[category]["embeddings"]   # [N, D]
    paths  = library[category]["paths"]
    tracks = library[category]["tracks"]

    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    e_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    scores = e_norm @ q_norm                   # [N]

    if exclude_track:
        for i, t in enumerate(tracks):
            if t == exclude_track:
                scores[i] = -1.0

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(paths[i], float(scores[i]), tracks[i]) for i in top_idx]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True)
    p.add_argument("--cache_dir",    default="data/cache")
    p.add_argument("--ckpt_path",    default="checkpoints/best.pt")
    p.add_argument("--library_path", default="data/cache/library.pt")
    args = p.parse_args()
    build_library(args.data_root, args.cache_dir, args.ckpt_path, args.library_path)
