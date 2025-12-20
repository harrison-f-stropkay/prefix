from datasets import load_dataset


fw = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-100BT",
    split="train",
    cache_dir="./data/",
)
