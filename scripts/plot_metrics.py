import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSITE_TASKS = {"arc_easy", "hellaswag", "piqa", "winogrande"}
SMOOTH_BIN_SIZE = 5


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


def smooth_rows(rows: list[dict], bin_size: int = SMOOTH_BIN_SIZE) -> list[dict]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame = frame.sort_values(["model", "tokens_seen"])
    frame["bin_id"] = frame.groupby("model").cumcount() // bin_size
    grouped = frame.groupby(["model", "bin_id"], as_index=False).agg(
        {"tokens_seen": "mean", "value": "mean"}
    )
    return grouped[["model", "tokens_seen", "value"]].to_dict("records")


def build_composite(eval_rows: list[dict], metric: str) -> list[dict]:
    per_task: dict[tuple[str, int, str], float] = {}
    for row in eval_rows:
        if row["task"] not in COMPOSITE_TASKS:
            continue
        if row["metric"] != metric:
            continue
        key = (row["model"], row["tokens_seen"], row["task"])
        if key not in per_task:
            per_task[key] = row["value"]

    per_step: dict[tuple[str, int], list[float]] = {}
    for (model, tokens_seen, _task), value in per_task.items():
        per_step.setdefault((model, tokens_seen), []).append(value)

    composite_rows: list[dict] = []
    for (model, tokens_seen), values in per_step.items():
        if len(values) != len(COMPOSITE_TASKS):
            continue
        composite_rows.append(
            {
                "model": model,
                "tokens_seen": tokens_seen,
                "value": sum(values) / len(values),
            }
        )
    return composite_rows


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
        train_frame["value"] = train_frame["loss"]
        train_rows = train_frame[["model", "tokens_seen", "value"]].to_dict("records")

    metrics = [
        ("train_loss", smooth_rows(train_rows)),
        ("composite_norm", smooth_rows(build_composite(eval_rows, "acc"))),
        (
            "charbench_norm",
            smooth_rows(
                [r for r in eval_rows if r["task"] == "charbench" and r["metric"] == "acc"]
            ),
        ),
        ("composite_acc_norm", smooth_rows(build_composite(eval_rows, "acc_norm"))),
        (
            "charbench_acc_norm",
            smooth_rows(
                [r for r in eval_rows if r["task"] == "charbench" and r["metric"] == "acc_norm"]
            ),
        ),
    ]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
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
        errorbar = None
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
