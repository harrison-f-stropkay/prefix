import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_metrics(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_metric(metrics: dict, keys: list[str]) -> float | None:
    for key in keys:
        if key in metrics:
            value = metrics[key]
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def find_charbench_key(results: dict) -> str | None:
    for key in results:
        if key.startswith("charbench"):
            return key
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training + eval metrics by tokens_seen.")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "figures" / "metrics.png")
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not runs_dir.exists():
        raise SystemExit(f"Runs dir not found: {runs_dir}")

    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        records = read_metrics(metrics_path)
        step_to_tokens: dict[int, int] = {}
        for record in records:
            if record.get("type") == "train":
                step = int(record["step"])
                if step % 1000 != 0:
                    continue
                tokens_seen = int(record["tokens_seen"])
                step_to_tokens[step] = tokens_seen
                train_rows.append(
                    {
                        "model": run_dir.name,
                        "tokens_seen": tokens_seen,
                        "value": float(record["loss"]),
                    }
                )

        for record in records:
            if record.get("type") != "eval":
                continue
            step = int(record["step"])
            tokens_seen = step_to_tokens.get(step)
            if tokens_seen is None:
                continue
            eval_path = Path(record["path"])
            if not eval_path.is_absolute():
                eval_path = REPO_ROOT / eval_path
            if not eval_path.exists():
                continue
            results = load_json(eval_path).get("results") or {}

            eval_rows.append(
                {
                    "model": run_dir.name,
                    "tokens_seen": tokens_seen,
                    "metric": "arc_easy",
                    "value": resolve_metric(
                        results.get("arc_easy", {}),
                        ["acc_norm,none", "acc,none"],
                    ),
                }
            )
            eval_rows.append(
                {
                    "model": run_dir.name,
                    "tokens_seen": tokens_seen,
                    "metric": "hellaswag",
                    "value": resolve_metric(
                        results.get("hellaswag", {}),
                        ["acc_norm,none", "acc,none"],
                    ),
                }
            )
            eval_rows.append(
                {
                    "model": run_dir.name,
                    "tokens_seen": tokens_seen,
                    "metric": "piqa",
                    "value": resolve_metric(
                        results.get("piqa", {}),
                        ["acc_norm,none", "acc,none"],
                    ),
                }
            )
            eval_rows.append(
                {
                    "model": run_dir.name,
                    "tokens_seen": tokens_seen,
                    "metric": "winogrande",
                    "value": resolve_metric(
                        results.get("winogrande", {}),
                        ["acc_norm,none", "acc,none"],
                    ),
                }
            )
            charbench_key = find_charbench_key(results)
            if charbench_key:
                eval_rows.append(
                    {
                        "model": run_dir.name,
                        "tokens_seen": tokens_seen,
                        "metric": "charbench_exact_match",
                        "value": resolve_metric(
                            results.get(charbench_key, {}), ["exact_match,none"]
                        ),
                    }
                )

    composite_rows: list[dict] = []
    composite_bins: dict[tuple[str, int], list[float]] = {}
    for row in eval_rows:
        if row["metric"] not in {"arc_easy", "hellaswag", "piqa", "winogrande"}:
            continue
        value = row["value"]
        if value is None:
            continue
        key = (row["model"], row["tokens_seen"])
        composite_bins.setdefault(key, []).append(value)
    for (model, tokens_seen), values in composite_bins.items():
        if len(values) != 4:
            continue
        composite_rows.append(
            {
                "model": model,
                "tokens_seen": tokens_seen,
                "metric": "lm_eval_composite",
                "value": sum(values) / 4.0,
            }
        )

    composite_smoothed: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    for row in composite_rows:
        grouped.setdefault(row["model"], []).append(row)
    for model, rows in grouped.items():
        rows.sort(key=lambda r: r["tokens_seen"])
        for i in range(0, len(rows), 5):
            chunk = rows[i : i + 5]
            if len(chunk) < 5:
                continue
            avg_tokens = sum(item["tokens_seen"] for item in chunk) / 5.0
            avg_value = sum(item["value"] for item in chunk) / 5.0
            composite_smoothed.append(
                {
                    "model": model,
                    "tokens_seen": avg_tokens,
                    "value": avg_value,
                }
            )

    metrics = [
        ("train_loss", train_rows),
        ("lm_eval_composite", composite_rows),
        ("lm_eval_composite_smoothed", composite_smoothed),
        ("charbench_exact_match", [r for r in eval_rows if r["metric"] == "charbench_exact_match"]),
    ]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.flatten()

    for idx, (title, rows) in enumerate(metrics):
        ax = axes[idx]
        if not rows:
            ax.set_title(f"{title} (no data)")
            ax.axis("off")
            continue
        frame = pd.DataFrame(rows).dropna(subset=["value"])
        if frame.empty:
            ax.set_title(f"{title} (no data)")
            ax.axis("off")
            continue
        sns.lineplot(
            data=frame,
            x="tokens_seen",
            y="value",
            hue="model",
            ax=ax,
            linewidth=1.6,
        )
        ax.set_title(title)
        ax.set_xlabel("tokens_seen")
        ax.set_ylabel("value")
        if idx != 0 and ax.get_legend() is not None:
            ax.get_legend().remove()

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
