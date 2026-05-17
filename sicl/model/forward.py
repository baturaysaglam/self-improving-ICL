"""
Batched forward pass and log-probability extraction for ICL prompts.

Stage 3: Run the model on batched embeddings to get output logits.
Stage 4: Extract log-probabilities for exemplar label tokens.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_with_embeddings(
    model: nn.Module,
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run a forward pass using embeddings instead of token IDs.
    
    Args:
        model: HuggingFace causal language model
        embeddings: Input embeddings of shape [batch_size, seq_len, hidden_dim]
        attention_mask: Optional attention mask of shape [batch_size, seq_len]
    
    Returns:
        logits: Output logits of shape [batch_size, seq_len, vocab_size]
        
    Note:
        For causal LMs, logits[:, j, :] predicts the token at position j+1.
        To get P(t_j | t_{<j}), use logits[:, j-1, :] and index with token_ids[j].
    """
    with torch.no_grad():
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,  # Don't cache for single forward pass
        )
    
    return outputs.logits


def extract_label_logprobs(
    logits: torch.Tensor,
    token_ids: List[int],
    token_spans: List[Tuple[int, int]],
) -> List[torch.Tensor]:
    """
    Extract log-probabilities for exemplar label tokens from model logits.
    
    For causal LMs, logits[:, j, :] predicts token at position j+1.
    Therefore, to get log P(token[j] | context), we use logits[:, j-1, :].
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        token_ids: Full sequence token IDs [seq_len]
        token_spans: List of (start, end) token index tuples for each exemplar label
                     (end is exclusive)
    
    Returns:
        List of tensors, one per exemplar. Each tensor has shape [batch_size, span_len]
        containing the log-probabilities of the true tokens in that exemplar's label.
        
    Note:
        For spans that start at position 0, tokens at position 0 are skipped because
        causal LMs cannot compute P(token[0] | context) without preceding context.
        This means the output tensor for such spans will have fewer elements than
        the span length. If the entire span is at position 0, an empty tensor is returned.
        
    Example:
        If token_spans = [(37, 38), (62, 64)] (first label is 1 token, second is 2 tokens):
        Returns: [Tensor[batch_size, 1], Tensor[batch_size, 2]]
        
        If token_spans = [(0, 3), (10, 12)] (first label starts at position 0):
        Returns: [Tensor[batch_size, 2], Tensor[batch_size, 2]]  # First span skips token 0
    
    Raises:
        ValueError: If any span has zero length
    """
    batch_size = logits.shape[0]
    
    # Convert to log-probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
    
    label_logprobs = []
    
    for span_idx, (start, end) in enumerate(token_spans):
        span_len = end - start
        
        if span_len == 0:
            raise ValueError(f"Span {span_idx} has zero length: ({start}, {end})")
        
        # Handle spans that start at position 0
        # For causal LM, we can't compute P(token[0] | context) - no preceding context
        # We skip token 0 and compute log-probs for tokens 1 onwards
        effective_start = max(start, 1)
        
        if effective_start >= end:
            # Entire span is at position 0, return empty tensor
            span_label_logprobs = torch.zeros(
                batch_size, 0, device=logits.device, dtype=logits.dtype
            )
        else:
            # Get the token IDs for the computable portion of the span
            span_token_ids = token_ids[effective_start:end]
            effective_span_len = len(span_token_ids)
            
            # For causal LM: logits at position j-1 predict token at position j
            # So for tokens at positions [effective_start, end), we need logits at [effective_start-1, end-1)
            # Shape: [batch_size, effective_span_len, vocab_size]
            span_log_probs = log_probs[:, effective_start-1:end-1, :]
            
            # Extract log-probs for the actual tokens
            # span_token_ids has shape [effective_span_len], need to gather along vocab dimension
            # Result shape: [batch_size, effective_span_len]
            token_indices = torch.tensor(span_token_ids, device=logits.device, dtype=torch.long)
            token_indices = token_indices.unsqueeze(0).expand(batch_size, -1)
            
            # Gather the log-probs for the true tokens
            span_label_logprobs = torch.gather(
                span_log_probs, 
                dim=2, 
                index=token_indices.unsqueeze(-1)  # [batch_size, effective_span_len, 1]
            ).squeeze(-1)  # [batch_size, effective_span_len]
        
        label_logprobs.append(span_label_logprobs)
    
    return label_logprobs
