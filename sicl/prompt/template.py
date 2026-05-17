"""
Prompt rendering and span localization for proxy computation.

This module provides functions to:
1. Render prompts from benchmark samples
2. Locate exemplar label positions (character-level spans) within rendered prompts

These functions are used by the proxy computation pipeline to convert
pre-extracted exemplar labels into token-level spans.

Note: This module does NOT extract labels from scratch. Labels are assumed to
already exist in the sample's 'exemplar_labels' field. This module only
determines WHERE those labels appear in the rendered prompt text.
"""

from typing import Dict, Any, Optional, List


# =============================================================================
# Prompt Rendering Functions
# =============================================================================

def render_dict_prompt(sample: Dict[str, Any]) -> str:
    """
    Render prompt for dict_search hash_string tasks.
    
    Format: "key : value\n" repeated, then "query_key :"
    """
    prompt = ""
    for k, v in sample["dict"].items():
        prompt += f"{k} : {v}\n"
    prompt += f"{sample['prompt']} :"
    return prompt


def render_standard_prompt(sample: Dict[str, Any]) -> str:
    """
    Render prompt using 'examples' + 'prompt' fields.
    
    This is the most common format: concatenate examples with the query prompt.
    """
    examples = sample.get("examples", "")
    return examples + sample["prompt"]


def render_exmaples_prompt(sample: Dict[str, Any]) -> str:
    """
    Render prompt for tasks with 'exmaples' typo (relation_analysis).
    
    Some benchmark data has a typo in the field name.
    """
    examples = sample.get("examples") or sample.get("exmaples", "")
    return examples + sample["prompt"]


def render_content_prompt(sample: Dict[str, Any]) -> str:
    """
    Render prompt for natural_language tasks using 'content' field.
    
    Format: content text followed by query prompt.
    """
    return sample["content"] + sample["prompt"]


# =============================================================================
# Template Configuration
# =============================================================================

# Standard config for Input/Output format tasks
_INPUT_OUTPUT_CONFIG = {
    "delimiter": "Output:",
    "label_terminator": "\n\n",
    "prompt_field": "examples",
    "render_fn": render_standard_prompt,
    "special_handler": None,
}

# Template configuration lookup table
# Key: (task, task_type) -> Value: parsing config dict
TEMPLATE_CONFIG: Dict[tuple, Dict[str, Any]] = {
    # =========================================================================
    # ACTUAL BENCHMARK TASK NAMES (from ICLEval JSON files)
    # =========================================================================
    
    # === Copying tasks ===
    ("dict_search", "hash_string"): {
        "delimiter": " : ",
        "label_terminator": "\n",
        "prompt_field": "dict",
        "render_fn": render_dict_prompt,
        "special_handler": "dict_search",
    },
    ("dict_search", "number-all_similar"): {
        "delimiter": " ⛱ ",
        "label_terminator": "\n",
        "prompt_field": "examples",
        "render_fn": render_standard_prompt,
        "special_handler": "sequence_matching",
    },
    ("dict_search", "number-half_similar"): {
        "delimiter": " ⛱ ",
        "label_terminator": "\n",
        "prompt_field": "examples",
        "render_fn": render_standard_prompt,
        "special_handler": "sequence_matching",
    },
    ("dict_search", "number-non_similar"): {
        "delimiter": " ⛱ ",
        "label_terminator": "\n",
        "prompt_field": "examples",
        "render_fn": render_standard_prompt,
        "special_handler": "sequence_matching",
    },
    ("natural_language", "hash_string_copying"): {
        "delimiter": None,
        "label_terminator": None,
        "prompt_field": "content",
        "render_fn": render_content_prompt,
        "special_handler": "hash_in_text",
    },
    
    # === Order tasks (Input/Output format) ===
    ("check_order", "character"): _INPUT_OUTPUT_CONFIG,
    ("check_order", "word"): _INPUT_OUTPUT_CONFIG,
    ("keep_order", "character"): _INPUT_OUTPUT_CONFIG,
    ("keep_order", "word"): _INPUT_OUTPUT_CONFIG,
    ("keep_order", "sentence"): _INPUT_OUTPUT_CONFIG,
    ("reversed_order", "character"): _INPUT_OUTPUT_CONFIG,
    ("reversed_order", "word"): _INPUT_OUTPUT_CONFIG,
    ("reversed_order", "sentence"): _INPUT_OUTPUT_CONFIG,
    ("specify_order", "character"): _INPUT_OUTPUT_CONFIG,
    ("specify_order", "word"): _INPUT_OUTPUT_CONFIG,
    
    # === Duplication tasks (Input/Output format) ===
    ("check_repeated_content", "character"): _INPUT_OUTPUT_CONFIG,
    ("check_repeated_content", "word"): _INPUT_OUTPUT_CONFIG,
    ("check_repeated_content", "sentence"): _INPUT_OUTPUT_CONFIG,
    ("remove_repeated_content", "character"): _INPUT_OUTPUT_CONFIG,
    ("remove_repeated_content", "word"): _INPUT_OUTPUT_CONFIG,
    ("find_repeated_content", "sentence"): _INPUT_OUTPUT_CONFIG,
    
    # === Format tasks ===
    ("format_convert", "normal"): _INPUT_OUTPUT_CONFIG,
    ("format_convert", "transfer"): _INPUT_OUTPUT_CONFIG,
    ("format_convert", "single"): {
        "delimiter": "Output:\n",
        "label_terminator": "\n\nInput:",
        "prompt_field": "examples",
        "render_fn": render_standard_prompt,
        "special_handler": "multiline_output",
    },
    ("format_convert", "multi"): {
        "delimiter": "Output:\n",
        "label_terminator": "\n\nInput:",
        "prompt_field": "examples",
        "render_fn": render_standard_prompt,
        "special_handler": "multiline_output",
    },
    ("format_convert", "mix"): {
        "delimiter": "Output:\n",
        "label_terminator": "\n\nInput:",
        "prompt_field": "examples",
        "render_fn": render_standard_prompt,
        "special_handler": "multiline_output",
    },
    
    # === Statistic/Relation tasks ===
    ("generate_statistic", "relation"): {
        "delimiter": "Output:",
        "label_terminator": "\n\nInput:",
        "prompt_field": "exmaples",
        "render_fn": render_exmaples_prompt,
        "special_handler": None,
    },
    
    # === Count/Navigation tasks ===
    ("count_or_navigation", "count-easy"): _INPUT_OUTPUT_CONFIG,
    ("count_or_navigation", "count-middle"): _INPUT_OUTPUT_CONFIG,
    ("count_or_navigation", "navigation-easy"): _INPUT_OUTPUT_CONFIG,
    ("count_or_navigation", "navigation-middle"): _INPUT_OUTPUT_CONFIG,
    
    # === List number tasks ===
    ("list_number", "list_number"): _INPUT_OUTPUT_CONFIG,
    
    # === Output format tasks (GSM8K style / MCQ) ===
    ("output_format", "output_format_01"): {
        "delimiter": "Response:",
        "label_terminator": "\n\nQuestion:",
        "prompt_field": "examples",
        "render_fn": render_standard_prompt,
        "special_handler": "cot_response",
    },
    ("output_format", "output_format_02"): {
        "delimiter": "Answer:",
        "label_terminator": "\n\nQuestion:",
        "prompt_field": "examples",
        "render_fn": render_standard_prompt,
        "special_handler": "mcq_answer",
    },
    ("output_format", "output_format_03"): {
        "delimiter": "Answer:",
        "label_terminator": "\n\nQuestion:",
        "prompt_field": "examples",
        "render_fn": render_standard_prompt,
        "special_handler": "mcq_answer",
    },
    
    # =========================================================================
    # ALIASES for data/tasks_data/ task names
    # =========================================================================
    ("dict_search_string", "hash_string"): {
        "delimiter": " : ",
        "label_terminator": "\n",
        "prompt_field": "dict",
        "render_fn": render_dict_prompt,
        "special_handler": "dict_search",
    },
    ("dict_search_number", "number-all_similar"): {
        "delimiter": " ⛱ ",
        "label_terminator": "\n",
        "prompt_field": "examples",
        "render_fn": render_standard_prompt,
        "special_handler": "sequence_matching",
    },
    ("natural_language_string", "hash_string_copying"): {
        "delimiter": None,
        "label_terminator": None,
        "prompt_field": "content",
        "render_fn": render_content_prompt,
        "special_handler": "hash_in_text",
    },
    ("classifier_order", "character"): _INPUT_OUTPUT_CONFIG,
    ("classifier_duplication", "character"): _INPUT_OUTPUT_CONFIG,
    ("classifier_format", "normal"): _INPUT_OUTPUT_CONFIG,
    ("generate_order", "character"): _INPUT_OUTPUT_CONFIG,
    ("generate_duplication", "character"): _INPUT_OUTPUT_CONFIG,
    ("count_navigation", "count-easy"): _INPUT_OUTPUT_CONFIG,
    ("relation_analysis", "relation"): {
        "delimiter": "Output:",
        "label_terminator": "\n\nInput:",
        "prompt_field": "exmaples",
        "render_fn": render_exmaples_prompt,
        "special_handler": None,
    },
}


def get_template_config(task: str, task_type: str) -> Dict[str, Any]:
    """
    Get template configuration for a given (task, task_type) pair.
    
    Looks up the parsing configuration needed to render prompts and locate
    label spans for a specific task type.
    
    Args:
        task: Task name from benchmark sample (e.g., "format_convert")
        task_type: Task type from benchmark sample (e.g., "normal")
    
    Returns:
        Config dict with keys: delimiter, label_terminator, prompt_field,
        render_fn, special_handler
    
    Raises:
        KeyError: If (task, task_type) combination is not in lookup table
    """
    key = (task, task_type)
    if key not in TEMPLATE_CONFIG:
        raise KeyError(
            f"Unknown (task, task_type) combination: {key}. "
            f"Available keys: {sorted(TEMPLATE_CONFIG.keys())}"
        )
    return TEMPLATE_CONFIG[key]


def get_template_config_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get template configuration directly from a benchmark sample.
    
    Extracts 'task' and 'task_type' fields from the sample and looks up
    the corresponding template configuration.
    
    Args:
        sample: Benchmark sample dict with 'task' and 'task_type' fields
    
    Returns:
        Config dict for this sample's template
    """
    task = sample["task"]
    task_type = sample["task_type"]
    return get_template_config(task, task_type)


def render_prompt(sample: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> str:
    """
    Render the full prompt text from a benchmark sample.
    
    Constructs the complete prompt string that would be fed to the model,
    including all exemplars and the query.
    
    Args:
        sample: Benchmark sample dict
        config: Optional pre-fetched config; if None, will be looked up
    
    Returns:
        Fully rendered prompt string
    """
    if config is None:
        config = get_template_config_from_sample(sample)
    return config["render_fn"](sample)


# =============================================================================
# Character-Level Span Extraction
# =============================================================================

def extract_exemplar_spans(
    prompt_text: str,
    config: Dict[str, Any],
    sample: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Locate exemplar labels within a rendered prompt (character-level positions).
    
    Given a rendered prompt and its template configuration, finds the character
    positions where each exemplar's label appears. This is needed to convert
    known labels (from 'exemplar_labels' field) into token-level spans.
    
    Args:
        prompt_text: Fully rendered prompt string
        config: Template config from get_template_config
        sample: Original sample dict (needed for some special handlers)
    
    Returns:
        List of dicts, each containing:
            - index: int (0-based exemplar index)
            - text: str (the label text)
            - char_start: int (start position, inclusive)
            - char_end: int (end position, exclusive)
    """
    handler = config.get("special_handler")
    
    if handler == "dict_search":
        if sample is None:
            raise ValueError("dict_search handler requires sample dict")
        return _extract_dict_search_spans(prompt_text, sample)
    
    elif handler == "hash_in_text":
        if sample is None:
            raise ValueError("hash_in_text handler requires sample dict")
        return _extract_hash_in_text_spans(prompt_text, sample)
    
    elif handler == "sequence_matching":
        return _extract_sequence_matching_spans(prompt_text, config)
    
    else:
        # Standard delimiter/terminator extraction
        return _extract_standard_spans(prompt_text, config)


def _extract_standard_spans(
    prompt_text: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract spans using standard delimiter/terminator pattern.
    
    Handles Input/Output, Question/Response, Question/Answer formats
    where labels appear between a delimiter and terminator.
    """
    delimiter = config["delimiter"]
    terminator = config["label_terminator"]
    
    spans = []
    
    search_start = 0
    while True:
        delim_pos = prompt_text.find(delimiter, search_start)
        if delim_pos == -1:
            break
        
        label_start = delim_pos + len(delimiter)
        
        # Handle optional space after delimiter
        if label_start < len(prompt_text) and prompt_text[label_start] == ' ':
            label_start += 1
        
        term_pos = prompt_text.find(terminator, label_start)
        
        if term_pos == -1:
            # This is the query (last delimiter) - no label after it
            break
        
        label_end = term_pos
        label_text = prompt_text[label_start:label_end]
        
        label_text_stripped = label_text.rstrip()
        if label_text_stripped:
            actual_end = label_start + len(label_text_stripped)
            spans.append({
                "index": len(spans),
                "text": label_text_stripped,
                "char_start": label_start,
                "char_end": actual_end,
            })
        
        search_start = term_pos + len(terminator)
    
    return spans


def _extract_dict_search_spans(
    prompt_text: str,
    sample: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract spans for dict_search hash_string format.
    
    Handles format: "key : value\n" repeated, then "query_key :"
    """
    spans = []
    delimiter = " : "
    
    dict_items = list(sample["dict"].items())
    
    search_start = 0
    for idx, (key, value) in enumerate(dict_items):
        key_pattern = f"{key}{delimiter}"
        key_pos = prompt_text.find(key_pattern, search_start)
        
        if key_pos == -1:
            continue
        
        label_start = key_pos + len(key_pattern)
        
        newline_pos = prompt_text.find("\n", label_start)
        if newline_pos == -1:
            label_end = len(prompt_text)
        else:
            label_end = newline_pos
        
        label_text = prompt_text[label_start:label_end]
        
        spans.append({
            "index": idx,
            "text": label_text,
            "char_start": label_start,
            "char_end": label_end,
        })
        
        search_start = label_end + 1
    
    return spans


def _extract_sequence_matching_spans(
    prompt_text: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract spans for dict_search number sequence format.
    
    Handles format: "seq ⛱ result\n" repeated
    """
    spans = []
    delimiter = config["delimiter"]
    terminator = config["label_terminator"]
    
    lines = prompt_text.split(terminator)
    
    char_pos = 0
    for idx, line in enumerate(lines):
        if delimiter not in line:
            char_pos += len(line) + len(terminator)
            continue
        
        delim_pos_in_line = line.rfind(delimiter)
        
        label_start_in_line = delim_pos_in_line + len(delimiter)
        label_text = line[label_start_in_line:]
        
        abs_label_start = char_pos + label_start_in_line
        abs_label_end = char_pos + len(line)
        
        if label_text:
            spans.append({
                "index": len(spans),
                "text": label_text,
                "char_start": abs_label_start,
                "char_end": abs_label_end,
            })
        
        char_pos += len(line) + len(terminator)
    
    return spans


def _extract_hash_in_text_spans(
    prompt_text: str,
    sample: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract spans for natural_language hash_string_copying format.
    
    Finds all occurrences of the full hash (prefix + suffix) in the content.
    """
    spans = []
    
    prefix = sample["prompt"]
    suffix = sample["label"]
    full_hash = prefix + suffix
    
    query_pos = prompt_text.rfind(prefix)
    
    while query_pos > 0:
        check_start = max(0, query_pos - len(suffix))
        if prompt_text[check_start:query_pos + len(prefix)] == full_hash:
            query_pos = prompt_text.rfind(prefix, 0, query_pos)
        else:
            break
    
    search_start = 0
    idx = 0
    while True:
        hash_pos = prompt_text.find(full_hash, search_start)
        if hash_pos == -1 or hash_pos >= query_pos:
            break
        
        spans.append({
            "index": idx,
            "text": full_hash,
            "char_start": hash_pos,
            "char_end": hash_pos + len(full_hash),
        })
        idx += 1
        
        search_start = hash_pos + len(full_hash)
    
    return spans


# =============================================================================
# Convenience Functions
# =============================================================================

def list_all_templates() -> list:
    """Return all registered (task, task_type) combinations."""
    return sorted(TEMPLATE_CONFIG.keys())


def extract_exemplar_spans_from_sample(
    sample: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> tuple:
    """
    Convenience function: render prompt and extract spans in one call.
    
    Args:
        sample: Benchmark sample dict
        config: Optional pre-fetched config
    
    Returns:
        (prompt_text, spans) tuple
    """
    if config is None:
        config = get_template_config_from_sample(sample)
    
    prompt_text = render_prompt(sample, config)
    spans = extract_exemplar_spans(prompt_text, config, sample)
    
    return prompt_text, spans


def validate_spans(prompt_text: str, spans: List[Dict[str, Any]]) -> List[str]:
    """
    Validate that all spans correctly index into the prompt text.
    
    Returns list of error messages (empty if all valid).
    """
    errors = []
    
    for span in spans:
        extracted = prompt_text[span["char_start"]:span["char_end"]]
        if extracted != span["text"]:
            errors.append(
                f"Span {span['index']}: extracted={repr(extracted)}, "
                f"expected={repr(span['text'])}"
            )
    
    return errors
