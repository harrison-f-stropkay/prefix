import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_BIN_STEPS = 100


def read_metrics(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training + eval metrics by tokens_seen.")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "figures" / "metrics.png")
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not runs_dir.exists():
        raise SystemExit(f"Runs dir not found: {runs_dir}")

    train_raw_rows: list[dict] = []
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
                tokens_seen = int(record["tokens_seen"])
                step_to_tokens[step] = tokens_seen
                train_raw_rows.append(
                    {
                        "model": run_dir.name,
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "loss": float(record["loss"]),
                    }
                )
            if record.get("type") == "eval":
                tokens_seen = record.get("tokens_seen")
                if tokens_seen is None:
                    step = record.get("step")
                    if step is None:
                        continue
                    tokens_seen = step_to_tokens.get(int(step))
                if tokens_seen is None:
                    continue
                if "task" not in record or "metric" not in record or "value" not in record:
                    continue
                eval_rows.append(
                    {
                        "model": run_dir.name,
                        "tokens_seen": int(tokens_seen),
                        "task": record["task"],
                        "metric": record["metric"],
                        "value": float(record["value"]),
                    }
                )

    train_rows: list[dict] = []
    if train_raw_rows:
        train_frame = pd.DataFrame(train_raw_rows)
        train_frame["bin_id"] = (train_frame["step"] // TRAIN_BIN_STEPS).astype(int)
        train_frame["tokens_seen"] = train_frame.groupby(["model", "bin_id"])[
            "tokens_seen"
        ].transform("mean")
        train_frame["value"] = train_frame["loss"]
        train_rows = train_frame[["model", "tokens_seen", "value"]].to_dict("records")

    composite_rows: list[dict] = []
    task_scores: dict[tuple[str, int, str], float] = {}
    for row in eval_rows:
        if row["task"] not in {"arc_easy", "hellaswag", "piqa", "winogrande"}:
            continue
        if row["metric"] not in {"acc_norm", "acc"}:
            continue
        key = (row["model"], row["tokens_seen"], row["task"])
        if row["metric"] == "acc_norm" or key not in task_scores:
            task_scores[key] = row["value"]

    composite_bins: dict[tuple[str, int], list[float]] = {}
    for (model, tokens_seen, _task), value in task_scores.items():
        composite_bins.setdefault((model, tokens_seen), []).append(value)
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
        (
            "charbench_exact_match",
            [
                r
                for r in eval_rows
                if r["task"].startswith("charbench") and r["metric"] == "exact_match"
            ],
        ),
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
        errorbar = "sd" if title == "train_loss" else None
        sns.lineplot(
            data=frame,
            x="tokens_seen",
            y="value",
            hue="model",
            ax=ax,
            linewidth=1.6,
            errorbar=errorbar,
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
