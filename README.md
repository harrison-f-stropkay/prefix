# Prefix-Aware LM Training

This repository tests whether training a Llama-3–style model (~500M parameters) with a custom **prefix-aware** label-smoothing objective can improve downstream performance.

Label smoothing is a regularization technique that replaces hard one-hot labels with softer targets by putting $1-\varepsilon$ probability on the correct class and spreading the remaining $\varepsilon$ evenly across the other $K-1$ classes (each gets $\varepsilon/(K-1)$). This technique often improves generalization in classification tasks.

In this repo, instead of distributing label-smoothing mass over all incorrect tokens, we assign it only to tokens whose decoded text is a **prefix** of the correct token’s decoded text, weighted toward longer prefixes. Example: if the gold token is “personality”, assign some mass to “person”.

We hypothesize that relative to traditional label smoothing, this approach will provide the model with stronger signal about the characters composing each token, ameliorating the canonical “strawberry” problem.

## Experimental Design

We sample our data from FineWeb-Edu, found [here](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Specifically, we gather data from the 100b token sample (`sample/100BT`), and we train our models to 20b tokens. To detect prefixes and string lengths, we use Python's `startswith` and `len` functions.

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

Let $\varepsilon = 0.1$. Let $y_t$ be the gold next token and $s=\text{decode}(y_t)$ its decoded string. Define the set of **proper-prefix tokens**

$$
P(s)=\{i:\ \text{decode}(i)\text{ is a strict prefix of } s\}.
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

Let $\varepsilon \in \{0.1, 1\}$. Let $R(s)$ be the set of prefixes of $s$ (note that $R$ differs from $P$ since $R$ includes $s$). Define a softmax over prefixes weighted by character length. Let $L_i = |\text{decode}(i)|$ for $i\in P(s)$ and

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
\tilde L_i=\frac{|\text{decode}(i)|}{|\text{decode}(y_t)|},\qquad
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

## Directory Structure

Decisive layout for this project (single dataset, multiple objectives, single-node 8×GPU). Keep training code importable, keep configs explicit, and keep artifacts out of git.

Conventions and reproducibility notes: `docs/project_conventions.md`.
Quickstart commands: `docs/how_to_run.md`.
Docs: `docs/experiments.md`, `docs/checkpointing.md`, `docs/reference.md`.

```text
.
├── src/
│   └── prefix/                # future Python package (no code yet)
├── configs/                   # YAML configs (checked in)
│   ├── data/                  # dataset location + packing params
│   ├── eval/                  # lm-eval-harness task sets
│   ├── model/                 # Llama-3–style model sizes/hparams
│   ├── objective/             # CE + prefix-aware objective variants + tau sweep
│   ├── runs/                  # top-level run specs (TOML)
│   └── train/                 # optimizer/schedule/ddp/checkpoint params
├── scripts/                   # runnable entrypoints (no code yet)
├── cluster/
│   └── runai/                 # Run:ai submission helpers
│       ├── spinup_workspace.sh # dev workspace (Jupyter, interactive debugging)
│       └── submit_train.sh     # training runs (clone repo in job)
├── docs/                      # project notes (see `docs/how_to_run.md`)
├── data/                      # local datasets/artifacts (gitignored)
└── runs/                      # training outputs (gitignored)
```

````

## Reproducability

### Data

```bash
uv run python scripts/download_fineweb_edu.py --data-config configs/data/fineweb_edu_pack2048.yaml
uv run python scripts/create_mds.py --data-config configs/data/fineweb_edu_pack2048.yaml
```
````
