# Prefix-Aware LM Training

This repository tests whether training a Llama-3–style model (~100M parameters) with a custom **prefix-aware** label-smoothing objective can improve downstream performance.

Label smoothing is a regularization technique that replaces hard one-hot labels with softer targets by putting $1-\varepsilon$ probability on the correct class and spreading the remaining $\varepsilon$ evenly across the other $K-1$ classes (each gets $\varepsilon/(K-1)$). This technique often improves generalization in classification tasks.

In this repo, we test teh following idea: instead of distributing label-smoothing mass over all incorrect tokens, assign label-smoothing mass only to tokens whose decoded text is a **prefix** of the correct token. Example: if the gold token is “personality”, assign some mass to “person”. In some sense, "person" is correct, since a generative model could produce "ality" in subsequent decoding steps, yielding the true text.

We hypothesize that relative to traditional label smoothing, this approach will provide the model with stronger signal about the characters composing each token, potentially ameliorating the canonical “strawberry” problem.

## Experimental Design

We sample our data from FineWeb-Edu, found [here](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Specifically, we gather data from the 100b token sample (`sample/100BT`), and we train our models to 2b tokens. To detect prefixes and string lengths, we operate in byte-level token space (i.e., tokenizer vocab pieces) to avoid lossy UTF-8 decoding.

All training runs are identical except for the training objective. We compare the following families of training objectives.

### 1. Standard One-hot

Let $V$ be the vocabulary and let $y_t \in V$ denote the gold next token at position $t$.  
The target distribution is the one-hot distribution

$$
q_i = \mathbf{1}[i = y_t].
$$

### 2. Standard Label Smoothing

Let $\varepsilon = 0.1$.  
The smoothed target distribution is

$$
q_i =
\begin{cases}
1 - \varepsilon & i = y_t \\
\varepsilon / (|V| - 1) & i \neq y_t .
\end{cases}
$$

### 3. Prefix-aware Label Smoothing (Simple)

Let $\varepsilon = 0.1$. Let $y_t$ be the gold next token and $s=\text{token}(y_t)$ its byte-level token string. Define the set of **proper-prefix tokens**

$$
P(s)=\{i:\ \text{token}(i)\text{ is a strict prefix of } s\}.
$$

We define the target distribution $q$ in two cases:

- If $P(s)=\emptyset$, we use the standard one-hot target: $q_i=\mathbf{1}[i=y_t]$.
- If $P(s)\neq\emptyset$,
  $$
  q_i =
  \begin{cases}
  1-\varepsilon & i=y_t \\
  \varepsilon/|P(s)| & i\in P(s) \\
  0 & \text{otherwise}.
  \end{cases}
  $$

### 4. Prefix-aware Label Smoothing (Softmax)

Let $\varepsilon \in \{0.1, 1\}$. Let $R(s)$ be the set of prefixes of $s$ (note that $R$ differs from $P$ since $R$ includes $s$). Define a softmax over prefixes weighted by character length. Let $L_i = |\text{token}(i)|$ for $i\in P(s)$ and

$$
w_i=\frac{\exp(L_i)}{\sum_{j\in P(s)}\exp(L_j)}, \qquad
q_i =
\begin{cases}
1-\varepsilon & i=y_t \\
\varepsilon\, w_i & i\in P(s) \\
0 & \text{otherwise}.
\end{cases}
$$

Note that when $\varepsilon = 1$, the target distribution is exactly the softmax over prefix lengths.

### 5. Prefix-aware Label Smoothing (Softmax, Normalized)

Same as (4), except the softmax uses normalized prefix length. Let

$$
\tilde L_i=\frac{|\text{token}(i)|}{|\text{token}(y_t)|},\qquad
w_i=\frac{\exp(\tilde L_i)}{\sum_{j\in P(s)}\exp(\tilde L_j)}.
$$

Then $q$ is defined exactly as in (4) with these weights:

$$
q_i =
\begin{cases}
1-\varepsilon & i=y_t \\
\varepsilon\, w_i & i\in P(s) \\
0 & \text{otherwise}.
\end{cases}
$$

<!-- ### Temperature Sweep

We evaluate the prefix-aware objectives under the following temperatures:

$\tau \in \{0.05,\ 0.1,\ 0.2,\ 0.5,\ 1.0,\ 2.0 \}$
 -->

### Evaluation

Intermittently during training, we evaluate models with EleutherAI’s `lm-eval-harness` on:

- `piqa`
- `winogrande`
- `arc_easy`
- `hellaswag`
- `charbench`

## Directory Structure

```text
.
├── src/
│   └── prefix/
├── configs/
│   ├── README.md
│   └── *.yaml
├── scripts/
├── cluster/
│   ├── spinup_workspace.sh
│   └── submit_train.sh
├── docs/
├── data/
└── runs/
```

## Reproducability

### Collate training data

```bash
uv run python scripts/download_fineweb_edu.py --run-config configs/ce_seed_0.yaml
uv run python scripts/create_mds.py --run-config configs/ce_seed_0.yaml
```

### Compute info about Unicde prefixes

```bash
uv run python scripts/plot_prefix_counts.py --hf-id meta-llama/Meta-Llama-3-8
```

### Submit training runs

```bash
bash cluster/submit_train.sh configs/ce_seed_0.yaml
bash cluster/submit_train.sh configs/label_smoothing_seed_0.yaml
bash cluster/submit_train.sh configs/prefix_simple_seed_0.yaml
bash cluster/submit_train.sh configs/prefix_norm_eps0p1_seed0.yaml
bash cluster/submit_train.sh configs/prefix_norm_eps1p0_seed0.yaml
bash cluster/submit_train.sh configs/prefix_unnorm_eps0p1_seed0.yaml
bash cluster/submit_train.sh configs/prefix_unnorm_eps1p0_seed0.yaml
```

### Plot metrics

```bash
uv run python scripts/plot_metrics.py
```
