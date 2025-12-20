# Prefix-Aware LM Training

This repository tests whether training a Llama-3–style model (~500M parameters) with a custom **prefix-aware** label-smoothing objective can improve downstream performance.

Label smoothing has been shown to be effective. Instead of smoothing some probability mass over all incorrect tokens, we assign that mass to tokens that are **prefixes of the correct token**, including the correct token. For example, if the gold token is “personality,” we assign some correctness to "person". This has some natural notion of correctness since in token generation, subsequent decoding steps could produce “ality.” We assign greater probabilities to longer prefixes.

We hypothesize that relative to traditional label smoothing, this approach will provide the model with stronger signal about the characters composing each token, ameliorating the canonical “strawberry” problem.

## Model Configuration

**Model Architecture:** Llama-3–style dense model with ~500m parameters

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

## Experimental Design

We compare three families of training objectives. All training runs are identical except for the training objective.

We sample our data from FineWeb-Edu, found [here](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Specifically, we gather data from the 100b token sample (`sample/100BT`), and we train our models to 20b tokens.
To make prefix detection trivial, we remove all non-ASCII characters.

### 1. Standard Cross-Entropy

Typical one-hot target.

### 2. Prefix-Softmax (Unnormalized)

Let:

- $y$ = gold token
- $s = \text{decode}(y)$ = token’s ASCII string
- $S$ = set of all vocabulary tokens whose decoded ASCII text is a **prefix** of $s$, including $s$ itself

$$
\text{score}(p)=
\begin{cases}
|p|, & \text{if } p \in S,\\
0, & \text{otherwise}.
\end{cases}
$$

To compute our target probability distribution, we take the softmax with temperature $\tau$ over the scores.

### 3. Prefix-Softmax (Normalized)

Identical to the unnormalized case, except that

$$
\text{score}(p)=
\begin{cases}
|p| / |s|, & \text{if } p \in S,\\
0, & \text{otherwise}.
\end{cases}
$$

Again, to compute our target probability distribution, we take the softmax with temperature $\tau$ over the scores.

### Temperature Sweep

We evaluate the prefix-aware objectives under the following temperatures:

$\tau \in \{0.05,\ 0.1,\ 0.2,\ 0.5,\ 1.0,\ 2.0 \}$

## Directory Structure

## Evaluation

We use EleutherAI's [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) for:

- HellaSwag
- PIQA
- Winogrande-debiased
- ARC-Easy
