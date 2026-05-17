"""Embedding optimization for ICL prompts via zeroth-order gradient ascent."""

import logging
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sicl.model.embeddings import get_embeddings
from sicl.prompt.tokenization import find_token_spans_from_sample
from sicl.prompt.template import render_prompt, get_template_config_from_sample
from .proxy import compute_zeroth_order_gradient, compute_symmetric_zeroth_order_gradient
from .grad import normalize_grad, clip_grad, project_cosine

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility across all relevant libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


@dataclass
class OptimizationConfig:
    """Configuration for embedding optimization."""
    N: int = 16
    mu: float = 0.001
    alpha: float = 0.6
    beta: float = 0.3
    gamma: float = 0.1
    num_iter: int = 250
    lr: float = 0.03
    cosine_sim_th: float = 0.2
    patience: int = 15
    snr_threshold: float = 0.0
    proxy_gate_threshold: float = 0.10
    grad_estimator: str = "forward"
    grad_strategy: str = "clip"
    grad_clip_value: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def optimize_sample(
    sample: Dict[str, Any],
    model: torch.nn.Module,
    tokenizer,
    embed_layer: torch.nn.Module,
    config: OptimizationConfig,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Optimize embeddings for a single sample to maximize the ICL proxy.

    Args:
        sample: Sample dictionary from the benchmark.
        model: Causal language model.
        tokenizer: Tokenizer for the model.
        embed_layer: Model's input embedding layer.
        config: Optimization hyperparameters.
        verbose: Print per-iteration progress.

    Returns:
        Tuple of (optimized_embeddings, stats_dict).
    """
    prompt_text, token_ids, token_spans, _ = find_token_spans_from_sample(sample, tokenizer)

    expected_prompt = render_prompt(sample, get_template_config_from_sample(sample))
    assert prompt_text == expected_prompt, "Prompt mismatch between tokenization and rendering."

    base_embeddings = get_embeddings(token_ids, embed_layer)
    embeddings = base_embeddings.clone()

    query_start_idx = token_spans[-1][1] if token_spans else None

    highest_proxy = 0.0
    highest_confidence_embeddings = base_embeddings.clone()
    best_iteration = 0
    initial_proxy = None
    cos_sim_after = 1.0

    iterations_without_improvement = 0
    final_iteration = config.num_iter
    nan_detected = False
    proxy_gated = False
    snr_gated = False
    low_snr_count = 0
    initial_snr = None
    proxy_history = []

    if config.grad_estimator == "symmetric":
        grad_fn = compute_symmetric_zeroth_order_gradient
    else:
        grad_fn = compute_zeroth_order_gradient

    for iter_i in range(config.num_iter):
        gradient, proxy, snr = grad_fn(
            model=model,
            embeddings=embeddings,
            batch_size=config.N,
            mu=config.mu,
            token_ids=token_ids,
            token_spans=token_spans,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            query_start_idx=query_start_idx,
        )

        proxy_history.append(proxy)

        if initial_proxy is None:
            initial_proxy = proxy

            if config.proxy_gate_threshold > 0 and initial_proxy < config.proxy_gate_threshold:
                proxy_gated = True
                final_iteration = 1
                highest_proxy = initial_proxy
                if verbose:
                    logger.info(
                        f"  Proxy gate: initial proxy {initial_proxy:.4f} "
                        f"< threshold {config.proxy_gate_threshold}, skipping optimization"
                    )
                break

        if initial_snr is None:
            initial_snr = snr

        if proxy > highest_proxy:
            highest_proxy = proxy
            highest_confidence_embeddings = embeddings.clone()
            best_iteration = iter_i + 1
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        if config.snr_threshold > 0 and snr < config.snr_threshold:
            low_snr_count += 1
            if low_snr_count >= config.patience or (config.patience == 0 and low_snr_count >= 3):
                final_iteration = iter_i + 1
                snr_gated = True
                if verbose:
                    logger.info(
                        f"  SNR gate at iteration {final_iteration}: "
                        f"{low_snr_count} consecutive low-SNR iterations "
                        f"(SNR={snr:.4f} < {config.snr_threshold})"
                    )
                break
            continue
        else:
            low_snr_count = 0

        if config.grad_strategy == "clip":
            gradient = clip_grad(gradient, max_norm=config.grad_clip_value)
        else:
            gradient = normalize_grad(gradient)
        embeddings = embeddings + config.lr * gradient

        if config.cosine_sim_th is not None and config.cosine_sim_th > 0.0:
            embeddings = project_cosine(embeddings, base_embeddings, config.cosine_sim_th)

        cos_sim_after = F.cosine_similarity(
            embeddings.flatten(), base_embeddings.flatten(), dim=0
        ).item()

        if torch.isnan(embeddings).any() or (isinstance(proxy, float) and proxy != proxy):
            final_iteration = iter_i + 1
            nan_detected = True
            logger.warning(f"  NaN detected at iteration {final_iteration}, stopping.")
            break

        if verbose:
            logger.info(
                f"  Iteration {iter_i+1}/{config.num_iter} — "
                f"Proxy: {proxy:.6f}, Cos Sim: {cos_sim_after:.6f}, SNR: {snr:.4f}"
            )

        if config.patience > 0 and iterations_without_improvement >= config.patience:
            final_iteration = iter_i + 1
            if verbose:
                logger.info(
                    f"  Early stopping at iteration {final_iteration} "
                    f"(no improvement for {config.patience} steps)"
                )
            break

    stats = {
        "uid": sample["uid"],
        "task": sample["task"],
        "task_type": sample["task_type"],
        "initial_proxy": initial_proxy,
        "final_proxy": highest_proxy,
        "proxy_improvement": highest_proxy - initial_proxy if initial_proxy else 0.0,
        "best_iteration": best_iteration,
        "final_cosine_similarity": cos_sim_after,
        "num_tokens": len(token_ids),
        "iterations_run": final_iteration,
        "early_stopped": final_iteration < config.num_iter,
        "nan_detected": nan_detected,
        "initial_snr": initial_snr,
        "snr_gated": snr_gated,
        "proxy_gated": proxy_gated,
        "proxy_history": proxy_history,
    }

    return highest_confidence_embeddings, stats
