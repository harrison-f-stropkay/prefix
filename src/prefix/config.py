"""Config loading and validation utilities."""

from pathlib import Path
from typing import Any

import yaml

LLAMA3_HF_ID = "meta-llama/Meta-Llama-3-8B"
LLAMA3_TOKENIZER_TYPE = "hf_llama3"


def load_run_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Run config must be a mapping, got {type(data).__name__}.")
    run_cfg = data.get("run") or {}
    run_name = run_cfg.get("name")
    if not run_name:
        raise ValueError("run.name is required in run config.")
    if path.suffix in {".yaml", ".yml"} and path.stem != run_name:
        raise ValueError(f"run.name must match filename stem ({path.stem!r}), got {run_name!r}.")
    assert_llama3_config(data)
    return data


def assert_llama3_config(config: dict[str, Any]) -> None:
    data_cfg = config.get("data") or {}
    tok_cfg = data_cfg.get("tokenizer") or {}
    hf_id = tok_cfg.get("hf_id")
    if hf_id != LLAMA3_HF_ID:
        raise ValueError(
            "Expected Llama 3 tokenizer config "
            f"(data.tokenizer.hf_id={LLAMA3_HF_ID}), got {hf_id!r}."
        )

    model_cfg = config.get("model") or {}
    model_tok = model_cfg.get("tokenizer") or {}
    tok_type = model_tok.get("type")
    if tok_type != LLAMA3_TOKENIZER_TYPE:
        raise ValueError(
            "Expected Llama 3 model tokenizer type "
            f"(model.tokenizer.type={LLAMA3_TOKENIZER_TYPE}), got {tok_type!r}."
        )
