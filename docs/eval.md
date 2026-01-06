# Evaluating a Local PyTorch (Llama-3–style) Pretrained Model with `lm-eval` (Recommended Path)

This note describes the **recommended, lowest-friction** approach for evaluating a local PyTorch checkpoint (500M params, Llama-3–style architecture) every _N_ steps using **`lm-eval` v0.4.9.2**.

## Why this approach

**Export each checkpoint to a Transformers/Hugging Face (HF) model directory** (config + weights + tokenizer), then run evaluation via the **`lm-eval` HF backend** (`model="hf"`). This avoids writing a custom `lm-eval` model wrapper and keeps evaluation scripts short and stable.

---

## High-level workflow

1. **Train** your model (Llama-3–style architecture).
2. At each checkpoint (e.g., every 10k steps), **export** to an HF directory:
   - `config.json`
   - model weights (`model.safetensors` preferred)
   - tokenizer artifacts (must match training)
3. Run `lm_eval.simple_evaluate(...)` on a small set of multiple-choice tasks.
4. Record per-task scores over steps.

---

# Suggested tasks:

- `piqa`
- `winogrande`
- `arc_easy`
- `hellaswag`

For a **5-minute wallclock** budget, you will likely need to limit samples; see _Sample caps_ below.

---

## Step 1: Export your PyTorch model to HF format

### Option A: You already use Transformers `LlamaForCausalLM` during training

If your training loop already uses `transformers` and your model is a `LlamaForCausalLM` (or equivalent),
you can export directly:

```python
model.save_pretrained(out_dir, safe_serialization=True)
tokenizer.save_pretrained(out_dir)
```

### Option B: You have a local `nn.Module` + `state_dict` (common)

The goal is to **instantiate** `transformers.LlamaForCausalLM(config)` and load your weights, then save.

#### 1) Build a `LlamaConfig` that matches training **exactly**

You must set all architecture-critical fields correctly. Common fields:

- `vocab_size`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads` (GQA)
- RoPE-related parameters (`rope_theta`, scaling if used)
- `max_position_embeddings` (context length)

Example skeleton (fill with your real values):

```python
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import torch

config = LlamaConfig(
    vocab_size=YOUR_VOCAB_SIZE,
    hidden_size=YOUR_HIDDEN,
    intermediate_size=YOUR_FFN,
    num_hidden_layers=YOUR_LAYERS,
    num_attention_heads=YOUR_HEADS,
    num_key_value_heads=YOUR_KV_HEADS,   # important for GQA
    rms_norm_eps=1e-6,
    max_position_embeddings=YOUR_CTX,
    rope_theta=YOUR_ROPE_THETA,          # Llama-3 often uses large theta
    # add other fields you changed (attention bias, etc.)
)

hf_model = LlamaForCausalLM(config)
state = torch.load("/path/to/your_state_dict.pt", map_location="cpu")
hf_model.load_state_dict(state, strict=False)  # ideally strict=True once keys match

out_dir = "/path/to/exported_hf_checkpoint"
hf_model.save_pretrained(out_dir, safe_serialization=True)

tok = AutoTokenizer.from_pretrained("/path/to/your_tokenizer_dir", use_fast=True)
tok.save_pretrained(out_dir)
```

#### Key details / gotchas

- **Tokenizer must match training.** If vocab or special tokens differ, evaluation will be misleading.
- **Config must match training.** A mismatch can load “successfully” with `strict=False` but behave incorrectly.
- Prefer `strict=True` once your key naming is stable, to catch silent mistakes.
- Use `safe_serialization=True` to write `.safetensors`.
- If your training uses custom modules or nonstandard attention, ensure your exported model is still loadable by `transformers`.

---

## Step 2: Evaluate with the `lm-eval` Python API

### Minimal evaluator (single checkpoint)

```python
import lm_eval

TASKS = ["piqa", "winogrande", "arc_easy", "hellaswag"]

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=/path/to/exported_hf_checkpoint,dtype=bfloat16",
    tasks=TASKS,
    num_fewshot=0,
    device="cuda:0",
    batch_size="auto",
    limit=200,  # applies per task
)

print(results["results"])
```

### Sample caps for a ~5 minute budget

Start conservative; increase until you hit your wallclock target.

A reasonable first try for 500M:

- `piqa`: 800
- `winogrande`: 800
- `arc_easy`: 300
- `hellaswag`: 200

Because `lm-eval` `limit` is **global** per run, you have two options:

1. **Run tasks in separate calls** with different limits, or
2. Use one global limit (easiest) and accept that it under-samples some tasks.

Recommended: split into “cheap 2-option” and “heavier 4-option” buckets.

Example:

```python
import lm_eval

def run_eval(pretrained_dir, tasks, limit):
    return lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={pretrained_dir},dtype=bfloat16",
        tasks=tasks,
        num_fewshot=0,
        device="cuda:0",
        batch_size="auto",
        limit=limit,
    )

# Cheap / frequent
r1 = run_eval("/path/to/ckpt", ["piqa", "winogrande"], limit=800)

# Heavier
r2 = run_eval("/path/to/ckpt", ["arc_easy", "hellaswag"], limit=200)
```

### “Lowest perplexity wins” for multiple-choice

For multiple-choice tasks, `lm-eval` scores each option by **log-likelihood** (effectively per-token cross-entropy) and chooses the best option.
This corresponds to “pick the answer with the lowest normalized loss/perplexity” for that option.

---

## Step 3: Run every 10k steps

If your checkpoints are stored like:

- `/checkpoints/step_10000/`
- `/checkpoints/step_20000/`
- ...

then:

1. export HF checkpoint to the same dir (or to a parallel export dir), and
2. run evaluation.

Skeleton:

```python
from pathlib import Path
import json
import lm_eval

def evaluate_step(hf_dir: str, step: int, out_json: str):
    tasks_fast = ["piqa", "winogrande"]
    tasks_heavy = ["arc_easy", "hellaswag"]

    r_fast = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_dir},dtype=bfloat16",
        tasks=tasks_fast,
        num_fewshot=0,
        device="cuda:0",
        batch_size="auto",
        limit=800,
    )
    r_heavy = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_dir},dtype=bfloat16",
        tasks=tasks_heavy,
        num_fewshot=0,
        device="cuda:0",
        batch_size="auto",
        limit=200,
    )

    payload = {
        "step": step,
        "fast": r_fast["results"],
        "heavy": r_heavy["results"],
        "versions": r_fast.get("config", {}),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2))

for step in range(10_000, 200_001, 10_000):
    hf_dir = f"/checkpoints/step_{step}"
    evaluate_step(hf_dir, step, f"/evals/step_{step}.json")
```

---

## Recommended evaluation settings (for stability)

- `num_fewshot=0` (faster, lower variance).
- Use a fixed `limit` and keep it the same across steps for comparability.
- Keep `batch_size="auto"` unless you observe instability; then pin an int.
- Always record:
  - `lm_eval` version, `transformers` version
  - tasks and limits
  - dtype/device
  - git commit of your training code (optional but useful)

---

## Notes on “validation” for one-epoch training

Even if you train for one epoch, the **training data is not held out**: the model has been optimized on it.
For tracking training progress, it is still useful to keep a small **held-out** slice of your pretraining mixture to compute loss/perplexity that is not biased by training.

However, the benchmark tasks above are naturally held out and are suitable for measuring generalization during training.

---

## Troubleshooting checklist

### Model loads but scores are nonsense

- Wrong tokenizer (vocab/special tokens differ from training).
- Config mismatch (especially `num_key_value_heads`, RoPE settings, vocab size).
- Accidentally using an instruction/chat template for a pretrained-only model.

### Evaluation is too slow

- Reduce `limit` (biggest lever).
- Ensure bf16/fp16 is used (`dtype=bfloat16` or `dtype=float16`).
- Use fewer tasks per run or evaluate heavy tasks less frequently (e.g., every 20k).

### Keys don’t match HF `LlamaForCausalLM`

- Write a one-time key remapping function to match HF naming, then switch to `strict=True`.
- Once matched, exporting becomes routine and robust.

---

## Reference: recommended default task cadence

- Every 10k: `piqa` + `winogrande` (higher limits)
- Every 10k or 20k: `arc_easy` + `hellaswag` (lower limits)

This pattern tends to give useful signal early without exceeding tight wallclock budgets.
