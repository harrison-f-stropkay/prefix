import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import yaml
from streaming import StreamingDataLoader, StreamingDataset
from transformers import AutoTokenizer


DATA_CONFIG_PATH = "configs/data/fineweb_edu_ascii_pack2048.yaml"
TRAIN_CONFIG_PATH = "configs/train/single_node_8gpu.yaml"

data_config = yaml.safe_load(Path(DATA_CONFIG_PATH).read_text(encoding="utf-8"))
train_config = yaml.safe_load(Path(TRAIN_CONFIG_PATH).read_text(encoding="utf-8"))

output_dir = Path(data_config["dir"])
batch_size = train_config["batch_size"]
tokenize_name = data_config["tokenizer"]["hf_id"]

tokenizer = AutoTokenizer.from_pretrained(tokenize_name)

dataset = StreamingDataset(
    local=str(output_dir),
    shuffle=True,
    batch_size=batch_size,
)

print(len(dataset))

dataloader = StreamingDataLoader(
    dataset=dataset,
    batch_size=batch_size,
)

print("state_dict: ", dataloader.state_dict())

# it = iter(dataloader)
# for batch_idx in range(num_batches):
#     try:
#         sample = next(it)
#     except StopIteration:
#         logging.warning("MDS ended after %d batches", batch_idx)
#         break

#     ids = sample["input_ids"][sample["input_ids"].shape[0] - 1]
#     text = tokenizer.decode(ids, skip_special_tokens=False)
#     print(f"--- last example in the batch {batch_idx} ---")
#     print(text)
#     print()
