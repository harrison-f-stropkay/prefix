from __future__ import annotations

from pathlib import Path

import torch

from prefix.config import load_run_config
from prefix.modeling import build_llama_model
from prefix.objectives import load_tokenizer

PROMPT = "The North won the Civil War because"
RUN_CONFIG = Path("configs/ce_seed_0.yaml")
CHECKPOINT = Path("runs/ce_seed_0/checkpoints/latest.pt")
MAX_NEW_TOKENS = 100


def main() -> None:
    config = load_run_config(RUN_CONFIG)
    tokenizer = load_tokenizer(config["data"]["tokenizer"]["hf_id"])
    model = build_llama_model(config["model"], vocab_size=len(tokenizer))
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # type: ignore[arg-type]

    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
