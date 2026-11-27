import torch


def msp(probs):
    """Maximum softmax probability (per sample)."""
    return probs.max(dim=1).values


def energy_score(logits, temperature=1.0):
    """Free energy score (higher indicates more likely out-of-distribution)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)
