# Configs

Configs define experiments and are checked in.

Format policy:

- Component configs under `configs/{data,model,train,objective,eval}/` are **YAML**.
- Top-level run specs under `configs/runs/` are **TOML** (simple “wiring” consumed by `cluster/runai/submit_train.sh`).

Each run should be representable as “shared YAML configs + one objective config”, wired together by a TOML run spec under `configs/runs/`.

## Run Config Format (TOML)

Cluster submission uses a single top-level run config so the CLI stays simple.

Minimum:

```toml
[run]
name = "ce_baseline"

[configs]
objective = "configs/objective/ce.yaml"
```

Optional:

```toml
[configs]
data = "configs/data/fineweb_edu_ascii_pack2048.yaml"
model = "configs/model/llama3_500m.yaml"
train = "configs/train/single_node_8gpu.yaml"
eval = "configs/eval/default.yaml"
```
