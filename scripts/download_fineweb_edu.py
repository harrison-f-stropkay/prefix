from datasets import load_dataset


fw = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-100BT",
    split="train",
    num_proc="4",
    cache_dir="./data/",
)
