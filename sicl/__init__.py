from .model import init_model, get_embeddings, forward_with_embeddings, extract_label_logprobs
from .prompt import render_prompt, get_template_config_from_sample, find_token_spans_from_sample
from .zoo import (
    compute_proxy,
    compute_zeroth_order_gradient,
    compute_symmetric_zeroth_order_gradient,
    normalize_grad,
    clip_grad,
    project_cosine,
    OptimizationConfig,
    optimize_sample,
    set_seed,
)
