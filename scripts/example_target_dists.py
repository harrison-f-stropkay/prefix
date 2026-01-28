from __future__ import annotations

import argparse
from pathlib import Path

from prefix.config import load_run_config
from prefix.objectives import build_lookup, build_target_distribution, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show which tokens receive nonzero mass for each training objective."
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=[" promoters", " personality", "the", "\n"],
        help="Token strings to inspect (use quotes for leading spaces).",
    )
    parser.add_argument(
        "--run-configs",
        nargs="*",
        default=None,
        help="Run config paths to inspect (defaults to configs/*.yaml).",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=50,
        help="Max nonzero tokens to display for prefix objectives.",
    )
    return parser.parse_args()


def resolve_run_configs(paths: list[str] | None) -> list[Path]:
    if paths:
        return [Path(p) for p in paths]
    return sorted(Path("configs").glob("*.yaml"))


def main() -> None:
    args = parse_args()
    run_configs = resolve_run_configs(args.run_configs)
    if not run_configs:
        raise SystemExit("No run configs found.")

    base_cfg = load_run_config(run_configs[0])
    tokenizer = load_tokenizer(base_cfg["data"]["tokenizer"]["hf_id"])
    lookup = build_lookup(tokenizer)
    vocab_size = len(tokenizer)

    for run_config in run_configs:
        config = load_run_config(run_config)
        objective = config.get("objective") or {}
        obj_type = objective.get("type", "cross_entropy")
        epsilon = float(objective.get("epsilon", 0.1))
        tau = float(objective.get("tau", 1.0))
        normalized = bool(objective.get("normalized", obj_type == "prefix_softmax_normalized"))
        proper_prefixes_only = objective.get("proper_prefixes_only")
        if obj_type in {"prefix_simple", "prefix_softmax", "prefix_softmax_normalized"}:
            if proper_prefixes_only is None:
                raise SystemExit(
                    f"{run_config} is missing objective.proper_prefixes_only for prefix objectives."
                )
            proper_prefixes_only = bool(proper_prefixes_only)
        else:
            proper_prefixes_only = bool(proper_prefixes_only or False)

        print(f"\n== {run_config.name} ({obj_type}, epsilon={epsilon}) ==")
        for token_str in args.tokens:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if not token_ids:
                print(f"\nToken {token_str!r}: no ids produced")
                continue
            print(f"\nToken {token_str!r} -> ids {token_ids}")
            for token_id in token_ids:
                dist = build_target_distribution(
                    lookup,
                    token_id,
                    obj_type,
                    proper_prefixes_only=proper_prefixes_only,
                    epsilon=epsilon,
                    tau=tau,
                    normalized=normalized,
                )
                if obj_type == "label_smoothing":
                    smooth = epsilon / max(vocab_size - 1, 1)
                    print(
                        f"  id={token_id}: nonzero=all tokens (vocab={vocab_size}) "
                        f"gold={1.0 - epsilon:.6f} others={smooth:.6f}"
                    )
                    continue
                nonzero = [(idx, w) for idx, w in enumerate(dist) if w > 0]
                nonzero.sort(key=lambda pair: pair[1], reverse=True)
                print(f"  id={token_id}: {len(nonzero)} nonzero tokens")
                for idx, weight in nonzero[: args.max_show]:
                    tok = tokenizer.convert_ids_to_tokens(idx)
                    print(f"    {idx:>6}  {weight:.6f}  {tok!r}")
                if len(nonzero) > args.max_show:
                    print(f"    ... ({len(nonzero) - args.max_show} more)")


if __name__ == "__main__":
    main()
