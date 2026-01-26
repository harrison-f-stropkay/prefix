import argparse
import json
import logging
from pathlib import Path

import torch

from prefix.config import load_run_config
from prefix.eval import evaluate_lm_harness
from prefix.modeling import build_llama_model
from prefix.objectives import load_tokenizer


class Args(argparse.Namespace):
    run_config: Path
    checkpoint: Path
    batch_size: int
    limit: int | None
    output: Path | None
    log_samples: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with lm-eval-harness.")
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Include sample generations in the lm-eval output JSON.",
    )
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    config = load_run_config(args.run_config)
    eval_cfg = config.get("eval") or {}
    tasks = (eval_cfg.get("lm_eval") or {}).get("tasks") or []
    if not tasks:
        raise ValueError("No lm-eval tasks defined in run config.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        log_samples=args.log_samples,
    )

    logging.info("lm-eval results: %s", results.get("results"))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
