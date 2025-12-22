# Prefix-Aware LM Training

This repository tests whether training a Llama-3–style model (~500M parameters) with a custom **prefix-aware** label-smoothing objective can improve downstream performance.

Instead of distributing label-smoothing mass over all incorrect tokens, we assign it only to tokens whose decoded text is a **prefix** of the correct token’s decoded text (including the correct token), weighted toward longer prefixes. Example: if the gold token is “personality”, assign some mass to “person”.

We hypothesize that relative to traditional label smoothing, this approach will provide the model with stronger signal about the characters composing each token, ameliorating the canonical “strawberry” problem.

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
- $t$ = any token

$$
\text{score}(t)=
\begin{cases}
\text{len}(t), & \text{if } t \in S,\\
0, & \text{otherwise}.
\end{cases}
$$

To compute our target probability distribution, we take the softmax with temperature $\tau$ over the scores.

### 3. Prefix-Softmax (Normalized)

Identical to the unnormalized case, except that

$$
\text{score}(t)=
\begin{cases}
\text{len}(t) / \text{len}(s), & \text{if } t \in S,\\
0, & \text{otherwise}.
\end{cases}
$$

Again, to compute our target probability distribution, we take the softmax with temperature $\tau$ over the scores.

### Temperature Sweep

We evaluate the prefix-aware objectives under the following temperatures:

$\tau \in \{0.05,\ 0.1,\ 0.2,\ 0.5,\ 1.0,\ 2.0 \}$

### Evaluation

We evaluate trained models with EleutherAI’s `lm-eval-harness` on:

- HellaSwag
- PIQA
- Winogrande-debiased
- ARC-Easy

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

## Reproducability

### Data

```bash
uv run python scripts/download_fineweb_edu.py --data-config ./configs/data/fineweb_edu_ascii_pack2048.yaml
uv run python scripts/create_mds.py --data-config ./configs/data/fineweb_edu_ascii_pack2048.yaml
```
