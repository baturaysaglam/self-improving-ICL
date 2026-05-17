"""
Token span localization for exemplar labels.

Converts character-level spans to token-level spans using tokenizer offset mapping.
"""

from typing import List, Dict, Any, Tuple, Optional
from transformers import PreTrainedTokenizerBase


def find_token_spans(
    prompt_text: str,
    char_spans: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    add_special_tokens: bool = False,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Convert character-level spans to token-level spans.
    
    Args:
        prompt_text: The full prompt text
        char_spans: List of dicts with 'char_start' and 'char_end' keys
                    (output from extract_exemplar_spans)
        tokenizer: HuggingFace tokenizer
        add_special_tokens: Whether to add special tokens during tokenization
    
    Returns:
        token_ids: List of token IDs for the full prompt
        token_spans: List of (start, end) token index tuples (end-exclusive)
    
    Raises:
        ValueError: If a span cannot be mapped to token indices
    """
    # Tokenize with offset mapping
    encoding = tokenizer(
        prompt_text,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
    )
    token_ids = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']  # List of (char_start, char_end) per token
    
    token_spans = []
    
    for span in char_spans:
        char_start = span['char_start']
        char_end = span['char_end']
        
        # Find token indices that cover this character range
        token_start = None
        token_end = None
        
        for token_idx, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
            # Skip special tokens (they have (0, 0) offset mapping)
            if tok_char_start == tok_char_end == 0 and token_idx > 0:
                continue
            
            # Check if this token overlaps with the character span
            # Overlap condition: tok_char_start < char_end AND tok_char_end > char_start
            if tok_char_start < char_end and tok_char_end > char_start:
                if token_start is None:
                    token_start = token_idx
                token_end = token_idx + 1  # Keep updating until we pass the span
        
        if token_start is None or token_end is None:
            raise ValueError(
                f"Could not map character span ({char_start}, {char_end}) to token indices. "
                f"Text: {repr(prompt_text[char_start:char_end])}"
            )
        
        token_spans.append((token_start, token_end))
    
    return token_ids, token_spans


def find_token_spans_from_sample(
    sample: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    config: Optional[Dict[str, Any]] = None,
    add_special_tokens: bool = False,
) -> Tuple[str, List[int], List[Tuple[int, int]], List[Dict[str, Any]]]:
    """
    Extract token spans directly from a sample dict.
    
    This is a convenience function that combines:
    1. Rendering the prompt
    2. Extracting character spans
    3. Converting to token spans
    
    Args:
        sample: Benchmark sample dict
        tokenizer: HuggingFace tokenizer
        config: Optional template config (will be looked up if not provided)
        add_special_tokens: Whether to add special tokens during tokenization
    
    Returns:
        prompt_text: The rendered prompt
        token_ids: List of token IDs
        token_spans: List of (start, end) token index tuples
        char_spans: List of character-level span dicts (for reference)
    """
    from .template import (
        get_template_config_from_sample,
        render_prompt,
        extract_exemplar_spans,
    )
    
    if config is None:
        config = get_template_config_from_sample(sample)
    
    prompt_text = render_prompt(sample, config)
    char_spans = extract_exemplar_spans(prompt_text, config, sample)
    token_ids, token_spans = find_token_spans(
        prompt_text, char_spans, tokenizer, add_special_tokens
    )
    
    return prompt_text, token_ids, token_spans, char_spans


def validate_token_spans(
    token_ids: List[int],
    token_spans: List[Tuple[int, int]],
    expected_labels: List[str],
    tokenizer: PreTrainedTokenizerBase,
    strict: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate that token spans decode to the expected label texts.
    
    Args:
        token_ids: Full list of token IDs
        token_spans: List of (start, end) token index tuples
        expected_labels: List of expected label strings
        tokenizer: Tokenizer for decoding
        strict: If True, require exact match; if False, allow whitespace differences
    
    Returns:
        Tuple of (all_valid, error_messages)
    """
    errors = []
    
    if len(token_spans) != len(expected_labels):
        errors.append(
            f"Number of spans ({len(token_spans)}) != number of labels ({len(expected_labels)})"
        )
        return False, errors
    
    for idx, ((start, end), expected) in enumerate(zip(token_spans, expected_labels)):
        # Decode the span
        span_tokens = token_ids[start:end]
        decoded = tokenizer.decode(span_tokens)
        
        # Compare
        if strict:
            match = decoded == expected
        else:
            # Allow leading/trailing whitespace differences (GPT-2 encodes spaces with tokens)
            match = decoded.strip() == expected.strip()
        
        if not match:
            errors.append(
                f"Span {idx}: decoded={repr(decoded)}, expected={repr(expected)}, "
                f"tokens={span_tokens}, token_range=({start}, {end})"
            )
    
    all_valid = len(errors) == 0
    return all_valid, errors


def debug_token_spans(
    prompt_text: str,
    token_ids: List[int],
    token_spans: List[Tuple[int, int]],
    char_spans: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """
    Print detailed debugging information about token spans.
    
    Args:
        prompt_text: The full prompt text
        token_ids: List of token IDs
        token_spans: List of (start, end) token index tuples
        char_spans: List of character-level span dicts
        tokenizer: Tokenizer for decoding
    """
    print(f"Total tokens: {len(token_ids)}")
    print(f"Number of spans: {len(token_spans)}")
    print("-" * 60)
    
    for idx, (token_span, char_span) in enumerate(zip(token_spans, char_spans)):
        start, end = token_span
        span_tokens = token_ids[start:end]
        decoded = tokenizer.decode(span_tokens)
        expected = char_span['text']
        
        print(f"Exemplar {idx}:")
        print(f"  Expected label: {repr(expected)}")
        print(f"  Char span: ({char_span['char_start']}, {char_span['char_end']})")
        print(f"  Token span: ({start}, {end})")
        print(f"  Token IDs: {span_tokens}")
        print(f"  Decoded: {repr(decoded)}")
        print(f"  Match: {decoded.strip() == expected.strip()}")
        
        # Show individual tokens
        print(f"  Individual tokens:")
        for i, tid in enumerate(span_tokens):
            tok_text = tokenizer.decode([tid])
            print(f"    [{start + i}] {tid} -> {repr(tok_text)}")
        print()
