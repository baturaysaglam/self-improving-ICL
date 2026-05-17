import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def normalize_grad(grad, eps=1e-8):
    """
    Normalize (w.r.t. L2 norm) the gradient per token position.
    Each token's gradient vector is scaled to unit L2 norm.

    After normalization, every token moves by exactly `lr` per step,
    regardless of the raw gradient magnitude.

    Args:
        grad: Gradient tensor of shape (T, hidden_dim)
        eps: Small constant for numerical stability

    Returns:
        Normalized gradient of shape (T, hidden_dim)
    """
    norms = grad.norm(p=2, dim=1, keepdim=True)  # shape (T, 1)
    grad_normalized = grad / (norms + eps)
    return grad_normalized


def clip_grad(grad, max_norm=1.0, eps=1e-8):
    """
    Clip the gradient per token position.
    Each token's gradient vector is scaled down to have at most `max_norm`
    L2 norm. Gradients with norm <= max_norm pass through unchanged.

    Unlike normalize_grad, this preserves the natural gradient magnitude
    when it is small (e.g., near the optimum), allowing the effective step
    size to decay as the optimization converges.

    Args:
        grad: Gradient tensor of shape (T, hidden_dim)
        max_norm: Maximum L2 norm per token. Default: 1.0
        eps: Small constant for numerical stability

    Returns:
        Clipped gradient of shape (T, hidden_dim)
    """
    norms = grad.norm(p=2, dim=1, keepdim=True)  # shape (T, 1)
    clip_coef = torch.clamp(max_norm / (norms + eps), max=1.0)
    return grad * clip_coef


def project_cosine(x: torch.Tensor, x_ref: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """
    Project x onto the surface where cosine_similarity(x, x_ref) = tau.
    If cos_sim(x, x_ref) < tau, move x toward x_ref until it reaches tau.
    """
    cos_sim = F.cosine_similarity(x.flatten(), x_ref.flatten(), dim=0)
    if cos_sim >= tau:
        return x

    logger.info(f"Projecting back onto cosine similarity ball to avoid semantic drift — cosine similarity: {cos_sim:.6f}")

    x_ref_norm_sq = torch.sum(x_ref ** 2)
    x_parallel = (torch.sum(x * x_ref) / x_ref_norm_sq) * x_ref
    x_perp = x - x_parallel

    x_parallel_norm = torch.norm(x_parallel)
    x_perp_norm = torch.norm(x_perp)

    if x_perp_norm > 1e-8:
        alpha = x_parallel_norm * torch.sqrt(torch.tensor(1.0 / tau**2 - 1.0, device=x.device)) / x_perp_norm
        x_proj = x_parallel + alpha * x_perp
    else:
        x_proj = x_parallel

    # Preserve original embedding norm (optional but recommended)
    x_proj = x_proj * (torch.norm(x) / (torch.norm(x_proj) + 1e-8))
    return x_proj
