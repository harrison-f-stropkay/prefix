import argparse
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
from typing import cast

import seaborn as sns
from matplotlib import pyplot as plt
from transformers import PreTrainedTokenizerFast

from prefix.objectives import build_lookup, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Analyze token prefix counts.")
    parser.add_argument("--hf-id", help="Hugging Face tokenizer id.")
    parser.add_argument("--tokenizer-file", help="Path to a tokenizer.json file.")
    args = parser.parse_args()

    if args.tokenizer_file:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_file)
    elif args.hf_id:
        tokenizer = load_tokenizer(args.hf_id)
    else:
        raise SystemExit("Provide --hf-id or --tokenizer-file.")

    lookup = build_lookup(tokenizer)
    vocab_size = len(tokenizer)
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "prefix_count_distribution.pdf"

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
    print(f"vocab size: {vocab_size}")
    print(f"min prefixes: {min_val}")
    print(f"max prefixes: {max_val}")
    print(f"mean prefixes: {mean_val:.2f}")
    print(f"median prefixes: {median_val}")
    print(f"mode prefixes: {mode_val}")

    N = 800
    print(f"\nTop {N} tokens by prefix count:")
    top_ids = sorted(range(vocab_size), key=lambda i: counts[i], reverse=True)[:N]
    for token_id in top_ids:
        prefix_ids, _ = lookup[token_id]
        prefix_strs = [token_strings[prefix_id] for prefix_id in prefix_ids]
        print(f"{token_id}:{token_strings[token_id]!r} -> {len(prefix_ids)}")
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
    print("\nPrefix lists for selected tokens:")
    for token_str in targets:
        token_ids = token_ids_by_string.get(token_str)
        if not token_ids:
            print(f"{token_str!r}: not found")
            continue
        for token_id in token_ids:
            prefix_ids, _ = lookup[token_id]
            prefix_strs = [token_strings[prefix_id] for prefix_id in prefix_ids]
            print(f"{token_id}:{token_str!r} -> {prefix_strs}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
