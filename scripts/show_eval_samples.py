from __future__ import annotations

import json
from pathlib import Path

import torch

from prefix.config import load_run_config
from prefix.eval import evaluate_lm_harness
from prefix.modeling import build_llama_model
from prefix.objectives import load_tokenizer

RUN_CONFIG = Path("configs/ce_seed_0.yaml")
CHECKPOINT = Path("runs/ce_seed_0/checkpoints/latest.pt")
TASKS: list[str] = []
LIMIT = 10
BATCH_SIZE = 1
MAX_SAMPLES = 10
OUTPUT = Path("/tmp/charbench_samples.json")


def main() -> None:
    config = load_run_config(RUN_CONFIG)
    eval_cfg = config.get("eval") or {}
    tasks = TASKS or (eval_cfg.get("lm_eval") or {}).get("tasks") or []
    if not tasks:
        raise SystemExit("No lm-eval tasks defined or passed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    tokenizer = load_tokenizer(config["data"]["tokenizer"]["hf_id"])
    model = build_llama_model(config["model"], vocab_size=len(tokenizer))
    model = model.to(device)  # type: ignore[arg-type]
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    results = evaluate_lm_harness(
        model,
        tokenizer,
        tasks,
        batch_size=BATCH_SIZE,
        device=device,
        limit=LIMIT,
        log_samples=True,
    )

    samples = results.get("samples") or {}
    for task in tasks:
        items = samples.get(task) or []
        print(f"\n=== {task} (showing up to {MAX_SAMPLES}) ===", flush=True)
        if not items:
            print("No samples returned for this task.", flush=True)
            continue
        for idx, sample in enumerate(items[:MAX_SAMPLES]):
            prompt = sample.get("doc", {}).get("query") or sample.get("doc", {}).get("question")
            target = sample.get("doc", {}).get("answer")
            output = sample.get("resps", [""])[0]
            print(f"\n[{idx}] prompt: {prompt!r}", flush=True)
            print(f"target: {target!r}", flush=True)
            print(f"output: {output!r}", flush=True)

    OUTPUT.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
