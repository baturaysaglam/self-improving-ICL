# Self-Improving In-Context Learning

This repository implements a method for improving in-context learning (ICL) performance by optimizing prompt embeddings. A self-supervised proxy returns the model's in-context task confidence as a bounded scalar value, which is then maximized with respect to the input embeddings using zeroth-order optimization. The method requires no white-box access, training, auxiliary data, or learned parameters.

## Method

Given a few-shot prompt, the method:

1. **Computes an ICL confidence proxy** from the exemplar label log-probabilities, combining per-exemplar confidence, pooled robustness, and information gain into a single scalar score.
2. **Estimates the gradient** via finite differences — samples $N$ noise directions, evaluates the proxy at each perturbation, and averages the scaled noise vectors.
3. **Updates the prompt embeddings** via gradient ascent with cosine similarity projection to prevent semantic drift from the original prompt.

## Quick Start

```bash
pip install torch transformers accelerate
```

Open [`cookbook.ipynb`](cookbook.ipynb) for a complete walkthrough that:
- Loads the ICLEval benchmark (12 tasks, 2040 samples)
- Identifies a sample the model (Llama 3.1-8B) answers incorrectly
- Runs the optimization, showing proxy improvement per iteration
- Demonstrates the model now predicts the correct answer

## Requirements

- Python 3.10+
- PyTorch 2.0+
- HuggingFace Transformers
- A GPU with at least 24 GB VRAM (for Llama 3.1-8B in FP16)
- Access to `meta-llama/Llama-3.1-8B` on HuggingFace

## Repository Structure

```
├── cookbook.ipynb          End-to-end demonstration notebook
├── data/
│   └── tasks_data/        ICLEval benchmark (12 tasks, 2040 samples)
└── sicl/                  Core library
    ├── model/             Model interface
    │   ├── core.py        Model initialization (init_model)
    │   ├── embeddings.py  Token ID to embedding conversion
    │   └── forward.py     Batched forward pass and log-prob extraction
    ├── prompt/            Prompt handling
    │   ├── template.py    Prompt rendering, template config, span extraction
    │   └── tokenization.py Character-to-token span mapping
    └── zoo/               Zeroth-order optimization
        ├── proxy.py       ICL confidence proxy + gradient estimators
        ├── grad.py        Gradient operations (clip, normalize, project)
        └── optimize.py    OptimizationConfig and optimize_sample()
```

<!-- ## Citation

```
[To be added upon publication]
``` -->
