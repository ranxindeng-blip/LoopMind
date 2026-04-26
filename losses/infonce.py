"""
Category-conditioned InfoNCE loss.
All 4 category losses are computed in one forward pass and summed.
"""
import torch
import torch.nn.functional as F

CATEGORIES = ["drums", "bass", "piano", "guitar"]


def infonce_loss(z_query: torch.Tensor, z_audio: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE for one category.
    z_query : [N, D] L2-normalized query embeddings
    z_audio : [N, D] L2-normalized audio embeddings
    Diagonal = positive pairs, off-diagonal = negatives.
    """
    sim   = torch.mm(z_query, z_audio.T) / temperature   # [N, N]
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_q = F.cross_entropy(sim,   labels)   # query → audio
    loss_a = F.cross_entropy(sim.T, labels)   # audio → query
    return (loss_q + loss_a) / 2


def category_infonce(z_queries: dict, z_audios: dict,
                     temperature: float = 0.07) -> torch.Tensor:
    """
    Sum InfoNCE losses across all categories that have at least 2 pairs.
    z_queries, z_audios : {category: [N_cat, D] tensor}
    """
    total = torch.tensor(0.0, device=next(iter(z_queries.values())).device)
    n_active = 0
    for cat in CATEGORIES:
        if cat not in z_queries or z_queries[cat].size(0) < 2:
            continue
        loss = infonce_loss(z_queries[cat], z_audios[cat], temperature)
        total += loss
        n_active += 1
    return total / max(n_active, 1)
