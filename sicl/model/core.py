"""Model initialization for ICL optimization."""

from transformers import AutoModelForCausalLM, AutoTokenizer


def init_model(model_args: dict) -> tuple:
    """Load a causal language model, tokenizer, and embedding layer.

    Args:
        model_args: Keyword arguments passed to AutoModelForCausalLM.from_pretrained.
                    Must include 'pretrained_model_name_or_path'.

    Returns:
        Tuple of (model, tokenizer, embed_layer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_args['pretrained_model_name_or_path'])
    model = AutoModelForCausalLM.from_pretrained(**model_args)
    model.eval()
    embed_layer = model.get_input_embeddings()

    return model, tokenizer, embed_layer
