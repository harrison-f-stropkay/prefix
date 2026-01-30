from __future__ import annotations

from pathlib import Path

from prefix.config import LLAMA3_HF_ID, load_run_config
import seaborn as sns
from matplotlib import pyplot as plt

from prefix.objectives import build_lookup, build_target_distribution, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENS = [" promoters", " personality", "the", "\n", " banana", " courage"]
MAX_SHOW = 50


def main() -> None:
    run_configs = sorted((REPO_ROOT / "configs").glob("*_fs5.yaml"))
    if not run_configs:
        raise SystemExit("No fs5 run configs found.")

    tokenizer = load_tokenizer(LLAMA3_HF_ID)
    lookup = build_lookup(tokenizer)
    vocab_size = len(tokenizer)

    lines: list[str] = []
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

        lines.append(f"\n== {run_config.name} ({obj_type}, epsilon={epsilon}) ==")
        for token_str in TOKENS:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if not token_ids:
                lines.append(f"\nToken {token_str!r}: no ids produced")
                continue
            lines.append(f"\nToken {token_str!r} -> ids {token_ids}")
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
                    lines.append(
                        f"  id={token_id}: nonzero=all tokens (vocab={vocab_size}) "
                        f"gold={1.0 - epsilon:.6f} others={smooth:.6f}"
                    )
                    continue
                nonzero = [(idx, w) for idx, w in enumerate(dist) if w > 0]
                nonzero.sort(key=lambda pair: pair[1], reverse=True)
                lines.append(f"  id={token_id}: {len(nonzero)} nonzero tokens")
                for idx, weight in nonzero[:MAX_SHOW]:
                    tok = tokenizer.convert_ids_to_tokens(idx)
                    lines.append(f"    {idx:>6}  {weight:.6f}  {tok!r}")
                if len(nonzero) > MAX_SHOW:
                    lines.append(f"    ... ({len(nonzero) - MAX_SHOW} more)")

    # Heatmaps: for each token, show probability mass over all prefixes.
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load objective configs.
    objectives = []
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
        objectives.append(
            {
                "name": run_config.stem,
                "type": obj_type,
                "epsilon": epsilon,
                "tau": tau,
                "normalized": normalized,
                "proper_prefixes_only": proper_prefixes_only,
            }
        )

    ncols = 2
    nrows = (len(TOKENS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), squeeze=False)

    for idx, token_str in enumerate(TOKENS):
        ax = axes[idx // ncols][idx % ncols]
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if not token_ids:
            ax.set_title(f"{token_str!r} (no ids)")
            ax.axis("off")
            continue
        token_id = token_ids[0]
        prefix_ids, _ = lookup[token_id]
        prefix_tokens = [tokenizer.convert_ids_to_tokens(pid) for pid in prefix_ids]
        if not prefix_ids:
            ax.set_title(f"{token_str!r} (no prefixes)")
            ax.axis("off")
            continue

        matrix = []
        col_labels = []
        for obj in objectives:
            dist = build_target_distribution(
                lookup,
                token_id,
                obj["type"],
                proper_prefixes_only=obj["proper_prefixes_only"],
                epsilon=obj["epsilon"],
                tau=obj["tau"],
                normalized=obj["normalized"],
            )
            row = [dist[pid] for pid in prefix_ids]
            matrix.append(row)
            col_labels.append(obj["name"])

        data = list(map(list, zip(*matrix)))
        sns.heatmap(
            data,
            cmap="viridis",
            xticklabels=col_labels,
            yticklabels=prefix_tokens,
            cbar=True,
            vmin=0.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_xlabel("objective")
        ax.set_ylabel("prefix token")
        ax.set_title(f"Prefix mass for token {token_str!r}")

    for j in range(len(TOKENS), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    out_pdf = figures_dir / "target_prefix_heatmaps.pdf"
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200)
    plt.close(fig)
    out_text = figures_dir / "example_target_dists.txt"
    out_text.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
