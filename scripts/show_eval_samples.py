from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch

from prefix.config import load_run_config
from prefix.eval import evaluate_lm_harness
from prefix.modeling import build_llama_model
from prefix.objectives import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-eval with log_samples and print model outputs."
    )
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(args.run_config)
    eval_cfg = config.get("eval") or {}
    tasks = args.tasks or (eval_cfg.get("lm_eval") or {}).get("tasks") or []
    if not tasks:
        raise SystemExit("No lm-eval tasks defined or passed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    tokenizer = load_tokenizer(config["data"]["tokenizer"]["hf_id"])
    model = build_llama_model(config["model"], vocab_size=len(tokenizer))
    model = model.to(device)  # type: ignore[arg-type]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    results = evaluate_lm_harness(
        model,
        tokenizer,
        tasks,
        batch_size=args.batch_size,
        device=device,
        limit=args.limit,
        log_samples=True,
    )

    samples = results.get("samples") or {}
    if not samples:
        print("No samples returned. Ensure log_samples=True and task supports samples.", flush=True)
        print(f"result keys: {list(results.keys())}", flush=True)
        return

    for task, items in samples.items():
        print(f"\n=== {task} (showing up to {args.max_samples}) ===", flush=True)
        for idx, sample in enumerate(items[: args.max_samples]):
            prompt = sample.get("doc", {}).get("query") or sample.get("doc", {}).get("question")
            target = sample.get("doc", {}).get("answer")
            output = sample.get("resps", [""])[0]
            print(f"\n[{idx}] prompt: {prompt!r}", flush=True)
            print(f"target: {target!r}", flush=True)
            print(f"output: {output!r}", flush=True)

    if args.output:
        args.output.write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
