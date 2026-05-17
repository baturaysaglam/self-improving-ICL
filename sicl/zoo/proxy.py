"""
ICL Confidence Proxy computation.

Implements the proxy components for assessing in-context learning performance:
- Component 1: Per-Exemplar Absolute Confidence
- Component 2: Per-Exemplar Robustness
- Component 3: Absolute Information Gain
"""

from typing import List, Optional, Tuple
import torch


def compute_per_exemplar_confidence(
    label_logprobs: List[torch.Tensor],
) -> torch.Tensor:
    """
    Compute per-exemplar absolute confidence (Component 1).
    
    For each exemplar i, computes:
        c_i = exp(mean(log_probs over tokens in exemplar i))
    
    This is equivalent to the geometric mean of true-token probabilities.
    
    Args:
        label_logprobs: List of N tensors, one per exemplar.
                        Each tensor has shape [batch_size, span_len_i] containing
                        log-probabilities of the true tokens in that exemplar's label.
                        Note: span_len_i may differ across exemplars.
    
    Returns:
        confidence: Tensor of shape [batch_size, N] where confidence[b, i] = c_i
                    for batch item b and exemplar i.
                    Range: (0, 1] for each element.
    
    Edge cases:
        - If an exemplar has span_len=0 (e.g., tokens at position 0 were skipped),
          its confidence is set to 0.0 (undefined/invalid).
    
    Example:
        >>> label_logprobs = [
        ...     torch.tensor([[-1.0, -2.0]]),  # exemplar 0: 2 tokens
        ...     torch.tensor([[-0.5]]),         # exemplar 1: 1 token
        ... ]
        >>> c = compute_per_exemplar_confidence(label_logprobs)
        >>> c.shape
        torch.Size([1, 2])
        >>> # c[0, 0] = exp((-1.0 + -2.0) / 2) = exp(-1.5)
        >>> # c[0, 1] = exp(-0.5 / 1) = exp(-0.5)
    """
    if len(label_logprobs) == 0:
        raise ValueError("label_logprobs cannot be empty")
    
    batch_size = label_logprobs[0].shape[0]
    num_exemplars = len(label_logprobs)
    device = label_logprobs[0].device
    dtype = label_logprobs[0].dtype
    
    # Allocate output tensor
    confidence = torch.zeros(batch_size, num_exemplars, device=device, dtype=dtype)
    
    for i, lp in enumerate(label_logprobs):
        span_len = lp.shape[1]
        
        if span_len == 0:
            # Empty span (e.g., all tokens were at position 0)
            # Set confidence to 0.0 to indicate invalid/undefined
            confidence[:, i] = 0.0
        else:
            # Compute mean log-prob across tokens: [batch_size]
            mean_logprob = lp.mean(dim=1)
            
            # Exponentiate to get confidence: c_i = exp(mean_logprob)
            confidence[:, i] = torch.exp(mean_logprob)
    
    return confidence


def compute_per_exemplar_robustness(
    label_logprobs: List[torch.Tensor],
    q: float = 0.1,
) -> torch.Tensor:
    """
    Compute per-exemplar robustness via low percentile (Component 2).
    
    For each exemplar i, computes:
        r_i = Quantile({exp(log_prob) for each token in exemplar i}, q)
    
    This measures "tail" confidence - the probability threshold below which
    the lowest q fraction of tokens fall.
    
    Args:
        label_logprobs: List of N tensors, one per exemplar.
                        Each tensor has shape [batch_size, span_len_i] containing
                        log-probabilities of the true tokens in that exemplar's label.
        q: Quantile level in [0, 1]. Default 0.1 (10th percentile).
           Lower values are more conservative (penalize low-probability tokens more).
    
    Returns:
        robustness: Tensor of shape [batch_size, N] where robustness[b, i] = r_i
                    for batch item b and exemplar i.
                    Range: [0, 1] for each element.
    
    Edge cases:
        - If an exemplar has span_len=0: robustness is set to 0.0 (undefined/invalid).
        - If an exemplar has span_len=1: the single probability is returned
          (quantile of a single element is that element).
    
    Implementation note:
        Uses torch.quantile with the default interpolation method ('linear').
        torch.quantile requires float32/float64, so conversion is done internally.
    
    Example:
        >>> label_logprobs = [
        ...     torch.tensor([[torch.log(torch.tensor(0.1)), 
        ...                    torch.log(torch.tensor(0.5)),
        ...                    torch.log(torch.tensor(0.9))]]),
        ... ]
        >>> r = compute_per_exemplar_robustness(label_logprobs, q=0.1)
        >>> # 10th percentile of [0.1, 0.5, 0.9] ≈ 0.1 (or interpolated)
    """
    if len(label_logprobs) == 0:
        raise ValueError("label_logprobs cannot be empty")
    
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    
    batch_size = label_logprobs[0].shape[0]
    num_exemplars = len(label_logprobs)
    device = label_logprobs[0].device
    dtype = label_logprobs[0].dtype
    
    # Allocate output tensor
    robustness = torch.zeros(batch_size, num_exemplars, device=device, dtype=dtype)
    
    for i, lp in enumerate(label_logprobs):
        span_len = lp.shape[1]
        
        if span_len == 0:
            # Empty span - set robustness to 0.0 (undefined/invalid)
            robustness[:, i] = 0.0
        else:
            # Convert log-probs to probs: p_j = exp(l_j)
            probs = torch.exp(lp)  # [batch_size, span_len]
            
            # Compute q-th quantile for each batch item
            # torch.quantile requires float32 or float64
            probs_float = probs.float()  # [batch_size, span_len]
            
            # torch.quantile(input, q, dim) computes quantile along dim
            # Shape: [batch_size]
            r_i = torch.quantile(probs_float, q, dim=1)
            
            # Convert back to original dtype
            robustness[:, i] = r_i.to(dtype)
    
    return robustness


def compute_information_gain(
    confidence: torch.Tensor,
) -> torch.Tensor:
    """
    Compute absolute information gain across exemplars (Component 3).
    
    Measures the average positive gain in confidence when moving from one
    exemplar to the next. Only counts improvements (negative changes are clamped to 0).
    
    For i = 2, ..., N:
        g_i = max(0, c_i - c_{i-1})
    
    Aggregate:
        G = (1/(N-1)) * sum(g_i)  if N >= 2
        G = 0                      if N = 1
    
    Args:
        confidence: Per-exemplar confidence tensor of shape [batch_size, N]
                    (output from compute_per_exemplar_confidence)
    
    Returns:
        gain: Tensor of shape [batch_size] where gain[b] = G for batch item b.
              Range: [0, 1] for each element.
    
    Edge cases:
        - N = 1: Returns 0 (no pairs to compare)
        - If c_i = 0 (invalid exemplar): The gain computation still proceeds,
          but gains involving invalid exemplars will be affected.
    
    Interpretation:
        G measures how much the model's per-exemplar confidence increases
        (only counting improvements) as additional exemplars are added.
        High G suggests the model is "learning" from the examples.
    
    Example:
        >>> confidence = torch.tensor([[0.1, 0.3, 0.2, 0.5]])  # 4 exemplars
        >>> G = compute_information_gain(confidence)
        >>> # g_2 = max(0, 0.3 - 0.1) = 0.2
        >>> # g_3 = max(0, 0.2 - 0.3) = 0.0  (decrease, clamped)
        >>> # g_4 = max(0, 0.5 - 0.2) = 0.3
        >>> # G = (0.2 + 0.0 + 0.3) / 3 = 0.167
    """
    if confidence.dim() != 2:
        raise ValueError(f"confidence must be 2D [batch_size, N], got shape {confidence.shape}")
    
    batch_size, num_exemplars = confidence.shape
    device = confidence.device
    dtype = confidence.dtype
    
    if num_exemplars == 0:
        raise ValueError("confidence cannot have 0 exemplars")
    
    if num_exemplars == 1:
        # Single exemplar: no pairs to compare, G = 0
        return torch.zeros(batch_size, device=device, dtype=dtype)
    
    # Compute consecutive differences: c_i - c_{i-1} for i = 2, ..., N
    # Shape: [batch_size, N-1]
    diffs = confidence[:, 1:] - confidence[:, :-1]
    
    # Clamp negative gains to 0: g_i = max(0, c_i - c_{i-1})
    gains = torch.clamp(diffs, min=0.0)
    
    # Average across pairs: G = (1/(N-1)) * sum(g_i)
    # Shape: [batch_size]
    G = gains.mean(dim=1)
    
    return G


def compute_pooled_robustness(
    label_logprobs: List[torch.Tensor],
    q: float = 0.1,
) -> torch.Tensor:
    """
    Compute robustness by pooling all label-token probabilities across all
    exemplars and taking a single quantile.

    Unlike compute_per_exemplar_robustness (which computes a per-exemplar
    quantile then averages), this pools every token probability from every
    exemplar into one set before computing the quantile. This avoids the
    degeneracy where single-token labels collapse the quantile to the
    confidence value itself.

    Args:
        label_logprobs: List of N tensors, one per exemplar.
                        Each tensor has shape [batch_size, span_len_i].
        q: Quantile level in [0, 1]. Default 0.1 (10th percentile).

    Returns:
        robustness: Tensor of shape [batch_size]. Range: [0, 1].
    """
    if len(label_logprobs) == 0:
        raise ValueError("label_logprobs cannot be empty")

    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")

    valid_lps = [lp for lp in label_logprobs if lp.shape[1] > 0]

    if not valid_lps:
        batch_size = label_logprobs[0].shape[0]
        device = label_logprobs[0].device
        dtype = label_logprobs[0].dtype
        return torch.zeros(batch_size, device=device, dtype=dtype)

    all_logprobs = torch.cat(valid_lps, dim=1)          # [batch_size, total_tokens]
    all_probs = torch.exp(all_logprobs)
    all_probs_float = all_probs.float()
    robustness = torch.quantile(all_probs_float, q, dim=1)  # [batch_size]

    return robustness.to(dtype=all_logprobs.dtype)


def compute_proxy(
    label_logprobs: List[torch.Tensor],
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
    q: float = 0.1,
) -> torch.Tensor:
    """
    Compute the combined ICL Confidence Proxy score.
    
    Combines three components:
        - C_hat: Mean per-exemplar confidence (Component 1 aggregated)
        - R_hat: Pooled robustness — quantile over all label tokens across
                 all exemplars (Component 2)
        - G: Absolute information gain (Component 3)
    
    Final score: f = alpha * C_hat + beta * R_hat + gamma * G
    
    Args:
        label_logprobs: List of N tensors, one per exemplar.
                        Each tensor has shape [batch_size, span_len_i] containing
                        log-probabilities of the true tokens in that exemplar's label.
        alpha: Weight for mean confidence C_hat. Default 0.6.
        beta: Weight for pooled robustness R_hat. Default 0.3.
        gamma: Weight for information gain G. Default 0.1.
        q: Quantile level for robustness computation. Default 0.1.
    
    Returns:
        proxy: Tensor of shape [batch_size] containing the combined proxy score.
               Range: [0, 1] when alpha + beta + gamma = 1 and all weights >= 0.
    
    Note:
        The weights alpha, beta, gamma should sum to 1 for the output to be in [0, 1],
        but this is not enforced to allow flexibility in experimentation.
    """
    # Component 1: Per-exemplar confidence -> aggregate to C_hat
    confidence = compute_per_exemplar_confidence(label_logprobs)  # [batch_size, N]
    C_hat = confidence.mean(dim=1)  # [batch_size]
    
    # Component 2: Pooled robustness
    R_hat = compute_pooled_robustness(label_logprobs, q=q)  # [batch_size]
    
    # Component 3: Information gain G (only computed when gamma > 0)
    if gamma > 0:
        G = compute_information_gain(confidence)  # [batch_size]
        proxy = alpha * C_hat + beta * R_hat + gamma * G
    else:
        proxy = alpha * C_hat + beta * R_hat
    
    return proxy


def _compute_snr(proxy_diff: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute the signal-to-noise ratio of proxy differences across the batch.

    SNR = |mean(proxy_diff)| / std(proxy_diff)

    When SNR << 1, the proxy differences are symmetric around zero, indicating
    the gradient estimate is dominated by noise rather than a reliable signal.

    Args:
        proxy_diff: Tensor of proxy differences [batch_size] or [num_pairs].
        eps: Small constant for numerical stability.

    Returns:
        SNR as a scalar float. Returns 0.0 if fewer than 2 samples.
    """
    if proxy_diff.numel() < 2:
        return 0.0
    return (proxy_diff.mean().abs() / (proxy_diff.std() + eps)).item()


def compute_zeroth_order_gradient(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    batch_size: int,
    mu: float,
    token_ids: List[int],
    token_spans: List[tuple],
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
    q: float = 0.1,
    query_start_idx: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute zeroth-order gradient estimate using finite differences.
    
    This function centralizes the entire gradient computation:
    1. Samples random noise and generates noisy embeddings
    2. Computes proxy values for both noisy and base embeddings
    3. Computes proxy differences
    4. Returns the gradient estimate, the base proxy value, and the
       gradient signal-to-noise ratio (SNR)
    
    Uses the formula:
        g ≈ (1 / (n * μ)) * Σ_i [f(x + μz_i) - f(x)] * z_i
    
    Which simplifies to:
        g = mean_i((proxy_diff[i] / μ) * z_i)
    
    Args:
        model: The language model for forward passes
        embeddings: Current embeddings [seq_len, hidden_dim]
        batch_size: Number of noisy samples to generate
        mu: Noise scale factor
        token_ids: Full sequence token IDs for log-prob extraction
        token_spans: List of (start, end) token index tuples for exemplar labels
        alpha: Weight for mean confidence in proxy computation
        beta: Weight for pooled robustness in proxy computation
        gamma: Weight for information gain in proxy computation
        q: Quantile level for robustness computation
        query_start_idx: If provided, zero out perturbation noise at positions
            >= this index. In a causal LM the proxy (computed from exemplar
            labels that precede the query) has zero true gradient w.r.t. query
            tokens; masking prevents the finite-sample noise at these positions
            from accumulating into a random walk.
        generator: Optional random generator for reproducibility
    
    Returns:
        gradient: Zeroth-order gradient estimate [seq_len, hidden_dim]
        proxy: Mean base proxy value (scalar) for monitoring optimization progress
        snr: Signal-to-noise ratio of the proxy differences across the batch.
             Values below ~1.0 indicate noise-dominated gradients.
    """
    from sicl.model.forward import forward_with_embeddings, extract_label_logprobs
    
    if mu == 0:
        raise ValueError("mu cannot be zero for gradient computation")
    
    seq_len, hidden_dim = embeddings.shape
    device = embeddings.device
    dtype = embeddings.dtype
    
    # --- Sample noise and compute noisy embeddings ---
    raw_noise = torch.randn(
        batch_size, seq_len, hidden_dim,
        device=device, dtype=dtype, generator=generator
    )

    if query_start_idx is not None and query_start_idx < seq_len:
        raw_noise[:, query_start_idx:, :] = 0.0

    base = embeddings.unsqueeze(0)
    noisy_embeddings = base + mu * raw_noise
    
    # --- Single forward pass: base + noisy concatenated ---
    all_embeddings = torch.cat([base, noisy_embeddings], dim=0)  # [N+1, seq_len, hidden_dim]
    logits = forward_with_embeddings(model, all_embeddings)
    label_logprobs = extract_label_logprobs(logits, token_ids, token_spans)
    proxy_all = compute_proxy(label_logprobs, alpha, beta, gamma, q)  # [N+1]
    
    # --- Split base and noisy proxy values ---
    proxy_base = proxy_all[0]
    proxy_noisy = proxy_all[1:]
    
    # --- Compute proxy difference ---
    proxy_diff = proxy_noisy - proxy_base  # [batch_size]
    
    # --- Gradient signal-to-noise ratio ---
    snr = _compute_snr(proxy_diff)
    
    # --- Compute gradient ---
    scale = (proxy_diff / mu).view(batch_size, 1, 1)
    scaled_noise = scale * raw_noise
    gradient = scaled_noise.mean(dim=0)
    
    return gradient, proxy_base.item(), snr


def compute_symmetric_zeroth_order_gradient(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    batch_size: int,
    mu: float,
    token_ids: List[int],
    token_spans: List[tuple],
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
    q: float = 0.1,
    query_start_idx: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute zeroth-order gradient estimate using symmetric (central-difference)
    sampling.

    For each of N/2 noise directions z_i, evaluates the proxy at both
    x + mu*z_i and x - mu*z_i, then uses the central-difference formula:

        g = (1 / (N/2)) * sum_i  [(f(x + mu*z_i) - f(x - mu*z_i)) / (2*mu)] * z_i

    Compared to the one-sided (forward-difference) estimator, this has two
    advantages:
      1. The f(x) baseline term cancels exactly, eliminating the dominant
         noise source (whose variance scales as f(x)^2 / mu^2).
      2. The bias is O(mu^2) instead of O(mu).

    Additionally, no separate base forward pass is needed — f(x) is estimated
    as the mean of (f(x + mu*z) + f(x - mu*z)) / 2 across all pairs, which
    is accurate to O(mu^2).

    Args:
        model: The language model for forward passes
        embeddings: Current embeddings [seq_len, hidden_dim]
        batch_size: Total number of forward passes (must be even).
                    Produces batch_size/2 symmetric pairs.
        mu: Noise scale factor
        token_ids: Full sequence token IDs for log-prob extraction
        token_spans: List of (start, end) token index tuples for exemplar labels
        alpha: Weight for mean confidence in proxy computation
        beta: Weight for pooled robustness in proxy computation
        gamma: Weight for information gain in proxy computation
        q: Quantile level for robustness computation
        query_start_idx: If provided, zero out perturbation noise at positions
            >= this index (see compute_zeroth_order_gradient).
        generator: Optional random generator for reproducibility

    Returns:
        gradient: Zeroth-order gradient estimate [seq_len, hidden_dim]
        proxy: Estimated base proxy value (scalar) for monitoring
               optimization progress. Computed as the mean of
               (f(x+mu*z) + f(x-mu*z)) / 2 across all pairs.
        snr: Signal-to-noise ratio of the proxy differences across pairs.
             Values below ~1.0 indicate noise-dominated gradients.

    Raises:
        ValueError: If mu is zero or batch_size is odd.
    """
    from sicl.model.forward import forward_with_embeddings, extract_label_logprobs

    if mu == 0:
        raise ValueError("mu cannot be zero for gradient computation")
    if batch_size % 2 != 0:
        raise ValueError(
            f"batch_size must be even for symmetric sampling, got {batch_size}"
        )

    num_pairs = batch_size // 2
    seq_len, hidden_dim = embeddings.shape
    device = embeddings.device
    dtype = embeddings.dtype

    # --- Sample N/2 noise directions ---
    raw_noise = torch.randn(
        num_pairs, seq_len, hidden_dim,
        device=device, dtype=dtype, generator=generator,
    )

    if query_start_idx is not None and query_start_idx < seq_len:
        raw_noise[:, query_start_idx:, :] = 0.0

    # --- Build symmetric batch ---
    base = embeddings.unsqueeze(0)
    pos_embeddings = base + mu * raw_noise
    neg_embeddings = base - mu * raw_noise

    paired = torch.stack(
        [pos_embeddings, neg_embeddings], dim=1,
    )
    all_embeddings = paired.reshape(batch_size, seq_len, hidden_dim)

    # --- Single forward pass for the entire batch ---
    logits = forward_with_embeddings(model, all_embeddings)
    label_logprobs = extract_label_logprobs(logits, token_ids, token_spans)
    proxy_all = compute_proxy(label_logprobs, alpha, beta, gamma, q)  # [batch_size]

    # --- Separate positive and negative proxy values ---
    proxy_all = proxy_all.view(num_pairs, 2)
    proxy_pos = proxy_all[:, 0]
    proxy_neg = proxy_all[:, 1]

    # --- Central-difference gradient ---
    proxy_diff = proxy_pos - proxy_neg
    scale = (proxy_diff / (2.0 * mu)).view(num_pairs, 1, 1)
    scaled_noise = scale * raw_noise
    gradient = scaled_noise.mean(dim=0)

    # --- Gradient signal-to-noise ratio ---
    snr = _compute_snr(proxy_diff)

    # --- Estimate f(x) for monitoring ---
    proxy_estimate = ((proxy_pos + proxy_neg) / 2.0).mean().item()

    return gradient, proxy_estimate, snr
