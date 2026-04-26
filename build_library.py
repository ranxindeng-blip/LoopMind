"""
Pre-compute audio embeddings for all stems → stem library.
Run once after training. Library is loaded at demo inference time.

Run location : browser Colab
Output stored : Google Drive (library_path)
"""
import os
import argparse
import numpy as np
import torch
from collections import defaultdict

from data.extract_pairs import extract_pairs
from data.features       import extract_and_cache, audio_to_mel_seq
from models.dual_encoder import DualEncoder, CATEGORIES


def build_library(data_root: str, cache_dir: str, ckpt_path: str,
                  max_tracks: int = 400,
                  library_path: str = "data/cache/library.pt"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    args  = ckpt["args"]
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

    library = {cat: {"embeddings": [], "paths": [], "tracks": [], "bpms": []}
               for cat in CATEGORIES}

    with torch.no_grad():
        for rec in records:
            track = rec["track"]
            bpm   = rec.get("bpm", 120.0)
            for cat in CATEGORIES:
                for audio_path, feat in features[track][cat].items():
                    x = torch.from_numpy(feat).unsqueeze(0).to(device)  # [1, T, 128]
                    z = model.encode_audio(x, cat).squeeze(0).cpu().numpy()
                    library[cat]["embeddings"].append(z)
                    library[cat]["paths"].append(audio_path)
                    library[cat]["tracks"].append(track)
                    library[cat]["bpms"].append(bpm)

    for cat in CATEGORIES:
        library[cat]["embeddings"] = np.stack(library[cat]["embeddings"])
        print(f"  {cat}: {len(library[cat]['paths'])} stems")

    os.makedirs(os.path.dirname(library_path) or ".", exist_ok=True)
    torch.save(library, library_path)
    print(f"\nLibrary saved to {library_path}")
    return library


def retrieve(query_emb: np.ndarray, library: dict, category: str,
             top_k: int = 3, exclude_track: str = None,
             query_bpm: float = None, bpm_tolerance: float = 0.20) -> list:
    """
    Cosine similarity search within one category.
    Optionally filter by BPM: only candidates within ±bpm_tolerance of query_bpm.
    Returns list of (audio_path, score, track, bpm).
    """
    embs   = library[category]["embeddings"]
    paths  = library[category]["paths"]
    tracks = library[category]["tracks"]
    bpms   = library[category].get("bpms", [120.0] * len(paths))

    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    e_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    scores = e_norm @ q_norm

    for i, (t, b) in enumerate(zip(tracks, bpms)):
        if exclude_track and t == exclude_track:
            scores[i] = -1.0
        # Suppress candidates whose BPM differs by more than tolerance
        if query_bpm is not None:
            ratio = max(b, query_bpm) / (min(b, query_bpm) + 1e-8)
            if ratio > 1.0 + bpm_tolerance:
                scores[i] = -1.0

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(paths[i], float(scores[i]), tracks[i], bpms[i]) for i in top_idx]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True)
    p.add_argument("--cache_dir",    default="data/cache")
    p.add_argument("--ckpt_path",    default="checkpoints/best.pt")
    p.add_argument("--max_tracks",   type=int, default=400)
    p.add_argument("--library_path", default="data/cache/library.pt")
    args = p.parse_args()
    build_library(args.data_root, args.cache_dir, args.ckpt_path,
                  args.max_tracks, args.library_path)
