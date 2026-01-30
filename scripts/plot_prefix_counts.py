from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
from typing import cast

import seaborn as sns
from matplotlib import pyplot as plt

from prefix.config import LLAMA3_HF_ID
from prefix.objectives import build_lookup, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    tokenizer = load_tokenizer(LLAMA3_HF_ID)

    lookup = build_lookup(tokenizer)
    vocab_size = len(tokenizer)
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "prefix_count_distribution.pdf"
    out_text = figures_dir / "prefix_count_distribution.txt"

    counts = [len(lookup[token_id][0]) for token_id in range(vocab_size)]
    mean_val = mean(counts)
    median_val = int(median(counts))
    freq = {}
    for count in counts:
        freq[count] = freq.get(count, 0) + 1
    mode_val = max(freq.items(), key=lambda item: item[1])[0]
    min_val = min(counts)
    max_val = max(counts)
    token_strings = [cast(str, tokenizer.convert_ids_to_tokens(i)) for i in range(vocab_size)]

    plt.figure(figsize=(7.0, 3.5))
    sns.histplot(counts, bins=range(min_val, max_val))
    plt.xlim(left=0)
    plt.yscale("log")
    plt.title("Prefix Count Distribution")
    plt.xlabel("Number of prefixes")
    plt.ylabel("Token count")
    lines: list[str] = []
    lines.append(f"vocab size: {vocab_size}")
    lines.append(f"min prefixes: {min_val}")
    lines.append(f"max prefixes: {max_val}")
    lines.append(f"mean prefixes: {mean_val:.2f}")
    lines.append(f"median prefixes: {median_val}")
    lines.append(f"mode prefixes: {mode_val}")

    N = 800
    lines.append(f"\nTop {N} tokens by prefix count:")
    top_ids = sorted(range(vocab_size), key=lambda i: counts[i], reverse=True)[:N]
    for token_id in top_ids:
        prefix_ids, _ = lookup[token_id]
        prefix_strs = [token_strings[prefix_id] for prefix_id in prefix_ids]
        lines.append(f"{token_id}:{token_strings[token_id]!r} -> {len(prefix_ids)}")
        # print(f"  prefixes: {prefix_strs}")

    token_ids_by_string: dict[str, list[int]] = {}
    for token_id, token_str in enumerate(token_strings):
        token_ids = token_ids_by_string.get(token_str)
        if token_ids is None:
            token_ids = []
            token_ids_by_string[token_str] = token_ids
        token_ids.append(token_id)

    targets = [" personal", " promoter", " tactful", " banana"]
    targets += [t.lstrip(" ") for t in targets]
    lines.append("\nPrefix lists for selected tokens:")
    for token_str in targets:
        token_ids = token_ids_by_string.get(token_str)
        if not token_ids:
            lines.append(f"{token_str!r}: not found")
            continue
        for token_id in token_ids:
            prefix_ids, _ = lookup[token_id]
            prefix_strs = [token_strings[prefix_id] for prefix_id in prefix_ids]
            lines.append(f"{token_id}:{token_str!r} -> {prefix_strs}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    out_text.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
