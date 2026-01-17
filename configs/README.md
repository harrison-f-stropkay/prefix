# Configs

Configs define experiments and are checked in.

Format policy:

- Single run configs live directly under `configs/` as **YAML**.
- Each run config is self-contained (no nested config references).
- `run.name` must match the filename stem and is used as the run directory name.

## Run Config Format (YAML)

Cluster submission uses a single top-level run config so the CLI stays simple.

Minimum:

```yaml
run:
  name: ce_seed_0
  type: ce
  seed: 0
objective:
  type: cross_entropy
```

Typical:

```yaml
run:
  name: ce_seed_0
  type: ce
  seed: 0
data:
  dataset:
    hf_id: HuggingFaceFW/fineweb-edu
    name: sample-100BT
    split: train
  tokenizer:
    hf_id: meta-llama/Meta-Llama-3-8B
  text_processing:
    num_proc: 64
    batch_size: 2048
  packing:
    sequence_length: 2048
  dir: ./data/mds/
model:
  architecture:
    hidden_size: 1536
    num_hidden_layers: 12
    num_attention_heads: 12
    num_key_value_heads: 4
    intermediate_size: 4096
    max_position_embeddings: 2048
    rope_theta: 500000
    rms_norm_eps: 1.0e-5
    tie_word_embeddings: true
  tokenizer:
    type: hf_llama3
train:
  distributed:
    world_size: 8
    backend: nccl
  checkpointing:
    enabled: true
    save_every_steps: 1000
    keep_last: 1
  per_gpu_batch_size: 8
  learning_rate: 2.0e-4
  weight_decay: 0.1
  adam_betas: [0.9, 0.95]
  adam_eps: 1.0e-8
  grad_clip: 1.0
eval:
  lm_eval:
    tasks:
      - piqa
      - winogrande
      - arc_easy
      - hellaswag
objective:
  type: cross_entropy
```
