This repository tests whether training a Llama-3–style model (~500M parameters) with a custom **prefix-aware** label-smoothing objective can improve downstream performance.

Label smoothing has been shown to be effective. Instead of smoothing some probability mass over all incorrect tokens, we assign that mass to **prefixes of the correct token**. For example, if the gold token is “personality,” we should not heavily punish the model for predicting “person,” since subsequent decoding steps can still produce “ality.” This also provides stronger signal about the characters composing each token, potentially ameliorating the canonical “strawberry” problem.

# Overview: Prefix-Aware LM Training

## 1. Model Configuration

**Model Architecture:** Llama-3–style dense model

- `hidden_size = 1536`
- `num_hidden_layers = 11`
- `num_attention_heads = 12`
- `num_key_value_heads = 4` (GQA)
- `intermediate_size = 4096`
- `max_position_embeddings = 2048`
- `rope_theta = 1e6`
- `rms_norm_eps = 1e-5`
- `tie_word_embeddings = True`

**Tokenizer:** HuggingFace Llama-3 tokenizer

## 2. Dataset Pipeline (ASCII-Only)

We use **FineWeb-Edu (HF)**. All text is normalized to ASCII so that each character is exactly one byte:

```python
text_ascii = text.encode("ascii", "ignore").decode("ascii")
```

This ensures prefix detection and prefix-length calculations are trivial.

Pipeline:

- Tokenize ASCII-normalized text
- Pack sequences to `seq_len = 2048`
- Persist using MosaicML `MDSWriter`; load with MosaicML Streaming during training

## 3. Training Objective

The repository supports three objectives.

### 3.1 Standard Cross-Entropy

One-hot target over the vocabulary.

### 3.2 Prefix-Softmax (Length-Normalized)

Let:

- $y$ = gold token
- $s = \text{decode}(y)$ = token’s ASCII string
- $L = \lvert s \rvert$ = length of the string
- $S_y$ = set of all vocabulary tokens whose decoded ASCII text is a **prefix** of $s$, including $s$ itself

For a prefix $p$ with length $k$:

$$\text{score}(p) = \dfrac{k}{L}$$

To compute our target probability distribution, we take the softmax with temperature $\tau$ over the scores, where the score is the **fraction** of $y$'s characters that appear in the prefix.

### 3.3 Prefix-Softmax (Unnormalized)

$$\text{score}(p) = k$$

To compute our target probability distribution, we take the softmax with temperature $\tau$ over the scores, where the score is the **number** of $y$'s characters that appear in the prefix.

## 4. Temperature Sweep

Evaluate the prefix-aware objectives under the following temperatures:

$\tau \in \{0.05,\ 0.1,\ 0.2,\ 0.5,\ 1.0,\ 2.0 \}$

## 5. Execution Plan

**Hardware:** 1 node, 8 GPUs (single-node DDP)

Each variant (objective × temperature) runs on its own node using all GPUs.

Keep constant across all runs:

- Total training tokens
- Global batch size (`micro_batch * grad_accum * gpus_per_node`)
- Optimizer, LR schedule
- Random seed

We use Python’s logging module for human-readable messages, but keep all numeric metrics in a separte csv file. Log training loss every N steps, validation loss every M steps, with each eval inits own csv line.

For checkpointing, every K steps, save a full “training state” dict, not just the model weights: include `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`, `global_step`, `tokens_seen`, and RNG states (PyTorch, CUDA, NumPy, Python).
Since we're using Run:ai, our runs can get preempted, so we should checkpoint fairly often and automatically initalize from the latest checkpoint.

## 6. Evaluation

Use EleutherAI **lm-eval-harness** (`lm-eval`) with:

- HellaSwag
- PIQA
- Winogrande-debiased
- ARC-Easy

## Gotchas

- Since we're using Run:ai, our runs can get preempted, so we should checkpoint fairly often and automatically initalize from the latest checkpoint.
