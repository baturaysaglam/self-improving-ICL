"""
Embedding extraction for ICL prompts.

Stage 2: Convert token IDs to embeddings.
"""

from typing import List, Optional
import torch
import torch.nn as nn


def get_embeddings(
    token_ids: List[int],
    embed_layer: nn.Embedding,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert token IDs to embeddings using the model's embedding layer.
    
    Args:
        token_ids: List of token IDs [seq_len]
        embed_layer: Model's input embedding layer
        device: Target device (defaults to embed_layer's device)
        dtype: Target dtype (defaults to embed_layer's dtype)
    
    Returns:
        embeddings: Tensor of shape [seq_len, hidden_dim]
    """
    if device is None:
        device = embed_layer.weight.device
    if dtype is None:
        dtype = embed_layer.weight.dtype
    
    token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
    
    with torch.no_grad():
        embeddings = embed_layer(token_tensor)  # [seq_len, hidden_dim]
    
    return embeddings.to(dtype=dtype)
