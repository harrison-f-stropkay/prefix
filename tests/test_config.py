from pathlib import Path

import pytest
import yaml

from prefix.config import LLAMA3_HF_ID, LLAMA3_TOKENIZER_TYPE, load_run_config


def build_run_config(
    name: str = "test",
    hf_id: str = LLAMA3_HF_ID,
    tok_type: str = LLAMA3_TOKENIZER_TYPE,
) -> dict:
    return {
        "run": {"name": name},
        "data": {"tokenizer": {"hf_id": hf_id}},
        "model": {"tokenizer": {"type": tok_type}},
    }


def write_config(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_load_run_config_valid(tmp_path: Path) -> None:
    path = tmp_path / "test.yaml"
    write_config(path, build_run_config(name="test"))
    config = load_run_config(path)
    assert config["run"]["name"] == "test"


def test_load_run_config_invalid_tokenizer(tmp_path: Path) -> None:
    path = tmp_path / "test.yaml"
    write_config(path, build_run_config(hf_id="not-llama"))
    with pytest.raises(ValueError):
        load_run_config(path)
