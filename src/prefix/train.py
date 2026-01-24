"""Training entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
from contextlib import nullcontext
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from streaming import StreamingDataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler

from prefix.config import load_run_config
from prefix.data import build_streaming_dataloader
from prefix.eval import run_lm_eval
from prefix.modeling import build_llama_model
from prefix.objectives import build_lookup, load_tokenizer

LOGGER = logging.getLogger(__name__)

OBJECTIVE_TYPES = {
    "cross_entropy",
    "label_smoothing",
    "prefix_simple",
    "prefix_softmax",
    "prefix_softmax_normalized",
}

PREFIX_OBJECTIVES = {
    "prefix_simple",
    "prefix_softmax",
    "prefix_softmax_normalized",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-config", required=True, type=Path)
    return parser.parse_args()


def configure_logging(log_path: Path | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if log_path is not None:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_adjust_shm_env() -> None:
    if os.environ.get("PREFIX_DISABLE_UCX_SHM") == "1":
        os.environ.setdefault("UCX_TLS", "^shm")
        os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
    if os.environ.get("PREFIX_DISABLE_NCCL_SHM") == "1":
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")


def init_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    _maybe_adjust_shm_env()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return rank, world, local_rank


def build_amp_context(device: torch.device) -> Any:
    if device.type == "cuda":
        # TF32 improves throughput for any FP32 matmuls without changing bf16 autocast behavior.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(device.index or 0)
        # H100s run best with bf16 autocast (range + throughput without fp16 underflow).
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def infer_run_dir(run_config: Path, runs_root: Path) -> Path:
    return runs_root / run_config.stem


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def capture_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("python") is not None:
        random.setstate(state["python"])


def write_metadata(output_dir: Path, config: dict[str, Any], run_config_path: Path) -> None:
    meta_dir = output_dir / "meta"
    config_dir = meta_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    run_yaml_path = config_dir / "run.yaml"
    if not run_yaml_path.exists():
        run_yaml_path.write_text(run_config_path.read_text(encoding="utf-8"), encoding="utf-8")

    run_config_path = config_dir / "run_config.json"
    if not run_config_path.exists():
        run_config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    git_path = meta_dir / "git.txt"
    if not git_path.exists():
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            commit = "unknown"
        git_path.write_text(f"git_commit={commit}\n", encoding="utf-8")

    env_path = meta_dir / "env.json"
    if not env_path.exists():

        def get_version(name: str) -> str | None:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return None

        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                "torch": get_version("torch"),
                "transformers": get_version("transformers"),
                "mosaicml-streaming": get_version("mosaicml-streaming"),
                "datasets": get_version("datasets"),
            },
            "cuda_available": torch.cuda.is_available(),
        }
        env_path.write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")


def prune_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    step_paths = list(checkpoint_dir.glob("step_*.pt"))
    if len(step_paths) <= keep_last:
        return

    def step_id(path: Path) -> int:
        stem = path.stem
        _, _, step_str = stem.partition("step_")
        return int(step_str)

    step_paths.sort(key=step_id)
    for path in step_paths[:-keep_last]:
        path.unlink(missing_ok=True)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataloader: Any,
    tokens_seen: int,
    *,
    keep_last: int,
    rank: int,
    world: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if world == 1:
        payload = {
            "step": step,
            "tokens_seen": tokens_seen,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "dataloader": dataloader.state_dict(),
            "rng": capture_rng_state(),
        }
        latest = checkpoint_dir / "latest.pt"
        torch.save(payload, latest)
        if keep_last > 0:
            step_path = checkpoint_dir / f"step_{step}.pt"
            torch.save(payload, step_path)
            prune_checkpoints(checkpoint_dir, keep_last)
        return

    if rank == 0:
        payload = {
            "step": step,
            "tokens_seen": tokens_seen,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "dataloader": dataloader.state_dict(),
        }
        latest = checkpoint_dir / "latest.pt"
        torch.save(payload, latest)
        if keep_last > 0:
            step_path = checkpoint_dir / f"step_{step}.pt"
            torch.save(payload, step_path)
            prune_checkpoints(checkpoint_dir, keep_last)

    torch.save(
        {
            "rng": capture_rng_state(),
        },
        checkpoint_dir / f"latest_rank{rank}.pt",
    )


def load_checkpoint_state(
    checkpoint_dir: Path,
    *,
    rank: int,
    world: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    latest = checkpoint_dir / "latest.pt"
    if world == 1 and latest.exists():
        state = load_checkpoint(latest)
        return state, state
    global_state = load_checkpoint(latest) if latest.exists() else None
    rank_state_path = checkpoint_dir / f"latest_rank{rank}.pt"
    rank_state = load_checkpoint(rank_state_path) if rank_state_path.exists() else None
    return global_state, rank_state


def build_prefix_weights(
    lookup: list[tuple[list[int], list[int]]],
    objective: dict[str, Any],
) -> list[tuple[list[int], list[float]]]:
    objective_type = objective["type"]
    tau = float(objective.get("tau", 1.0))
    normalized = bool(objective.get("normalized", False))
    vocab_size = len(lookup)
    weights: list[tuple[list[int], list[float]]] = [([], []) for _ in range(vocab_size)]

    if objective_type not in {
        "prefix_simple",
        "prefix_softmax",
        "prefix_softmax_normalized",
    }:
        return weights

    for token_id in range(vocab_size):
        prefix_ids, prefix_lengths = lookup[token_id]
        pairs = [
            (pid, plen)
            for pid, plen in zip(prefix_ids, prefix_lengths, strict=True)
            if pid != token_id
        ]
        if not pairs:
            continue
        ids = [pid for pid, _ in pairs]
        lens = [plen for _, plen in pairs]

        if objective_type == "prefix_simple":
            share = 1.0 / len(ids)
            weights[token_id] = (ids, [share for _ in ids])
            continue

        if normalized or objective_type == "prefix_softmax_normalized":
            denom = max(prefix_lengths) or 1
            lens = [plen / denom for plen in lens]

        scaled = [length / tau for length in lens]
        max_scaled = max(scaled)
        exp_sum = sum(math.exp(s - max_scaled) for s in scaled)
        probs = [math.exp(s - max_scaled) / exp_sum for s in scaled]
        weights[token_id] = (ids, probs)

    return weights


def build_prefix_tables(
    prefix_weights: list[tuple[list[int], list[float]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab_size = len(prefix_weights)
    max_len = max((len(ids) for ids, _ in prefix_weights), default=0)
    prefix_ids = torch.zeros((vocab_size, max_len), dtype=torch.int64)
    prefix_weights_tensor = torch.zeros((vocab_size, max_len), dtype=torch.float32)
    prefix_counts = torch.zeros(vocab_size, dtype=torch.int64)
    for token_id, (ids, weights) in enumerate(prefix_weights):
        if not ids:
            continue
        prefix_counts[token_id] = len(ids)
        prefix_ids[token_id, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
        prefix_weights_tensor[token_id, : len(weights)] = torch.tensor(weights, dtype=torch.float32)
    return prefix_ids, prefix_weights_tensor, prefix_counts


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective_type: str,
    epsilon: float,
    prefix_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    flat_labels = labels.reshape(-1)
    if objective_type == "cross_entropy":
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            flat_labels,
            reduction="mean",
        )
    if objective_type == "label_smoothing":
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            flat_labels,
            label_smoothing=epsilon,
        )
    if prefix_tables is None:
        raise RuntimeError("Prefix tables were not initialized.")
    log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
    prefix_ids, prefix_weights_tensor, prefix_counts = prefix_tables
    prefix_ids_batch = prefix_ids[flat_labels]
    prefix_weights_batch = prefix_weights_tensor[flat_labels]
    logp_true = log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    prefix_logp = log_probs.gather(1, prefix_ids_batch)
    prefix_term = (prefix_logp * prefix_weights_batch).sum(dim=1)
    has_prefix = prefix_counts[flat_labels] > 0
    return torch.where(
        has_prefix,
        -(1.0 - epsilon) * logp_true - epsilon * prefix_term,
        -logp_true,
    ).mean()


def build_dataloader(
    data_cfg: dict[str, Any],
    *,
    per_device_batch: int,
    shuffle: bool,
    seed: int,
) -> tuple[StreamingDataLoader, int]:
    data_dir = Path(data_cfg["dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    seq_len = int(data_cfg["packing"]["sequence_length"])
    streaming_cfg = data_cfg.get("streaming") or {}
    loader = build_streaming_dataloader(
        data_dir,
        per_device_batch,
        shuffle=shuffle,
        streaming_cfg=streaming_cfg,
        shuffle_seed=seed,
    )
    return loader, seq_len


def build_model_and_tokenizer(
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any]:
    tokenizer = load_tokenizer(data_cfg["tokenizer"]["hf_id"])
    model = build_llama_model(model_cfg, vocab_size=len(tokenizer)).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    return model, tokenizer


def build_optimizer_scheduler(
    model: torch.nn.Module,
    train_cfg: dict[str, Any],
    *,
    tokens_per_step: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    lr = float(train_cfg.get("learning_rate", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    adam_betas = train_cfg.get("adam_betas", [0.9, 0.95])
    if len(adam_betas) != 2:
        raise ValueError("train.adam_betas must have two values.")
    adam_eps = float(train_cfg.get("adam_eps", 1e-8))
    max_steps = int(train_cfg.get("max_steps", 100))
    max_tokens = int(train_cfg.get("max_tokens", 0))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    lr_scheduler = str(train_cfg.get("lr_scheduler", "cosine"))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(float(adam_betas[0]), float(adam_betas[1])),
        eps=adam_eps,
    )
    if max_steps > 0:
        total_steps = max_steps
    elif max_tokens > 0:
        total_steps = max(1, math.ceil(max_tokens / tokens_per_step))
    else:
        total_steps = 1
    scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler, total_steps


def resolve_objective(
    objective: dict[str, Any],
    *,
    tokenizer: Any,
    device: torch.device,
) -> tuple[str, float, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    objective_type = objective.get("type", "cross_entropy")
    if objective_type not in OBJECTIVE_TYPES:
        raise ValueError(f"Unknown objective type: {objective_type!r}")
    epsilon = float(objective.get("epsilon", 0.1))
    prefix_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if objective_type in PREFIX_OBJECTIVES:
        lookup = build_lookup(tokenizer)
        prefix_weights = build_prefix_weights(lookup, objective)
        prefix_tables = build_prefix_tables(prefix_weights)
        prefix_ids, prefix_weights_tensor, prefix_counts = prefix_tables
        prefix_tables = (
            prefix_ids.to(device),
            prefix_weights_tensor.to(device),
            prefix_counts.to(device),
        )
    return objective_type, epsilon, prefix_tables


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or infer_run_dir(args.run_config, Path("runs"))
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir / "train.log")
    config = load_run_config(args.run_config)

    rank, world, local_rank = init_dist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        device = torch.device("cuda", local_rank)
    amp_ctx = build_amp_context(device)

    train_cfg = config.get("train") or {}
    data_cfg = config.get("data") or {}
    model_cfg = config.get("model") or {}
    objective = config.get("objective") or {}
    run_cfg = config.get("run") or {}
    seed = int(run_cfg.get("seed", 0))
    set_global_seed(seed)

    per_device_batch = int(train_cfg.get("per_gpu_batch_size", 1))
    max_steps = int(train_cfg.get("max_steps", 100))
    max_tokens = int(train_cfg.get("max_tokens", 0))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    checkpoint_cfg = train_cfg.get("checkpointing") or {}
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", True))
    save_every = int(checkpoint_cfg.get("save_every_steps", 100))
    keep_last = int(checkpoint_cfg.get("keep_last", 0))
    shuffle = bool(train_cfg.get("shuffle", True))
    loader, seq_len = build_dataloader(
        data_cfg,
        per_device_batch=per_device_batch,
        shuffle=shuffle,
        seed=seed,
    )
    model, tokenizer = build_model_and_tokenizer(model_cfg, data_cfg, device=device)
    tokens_per_step = per_device_batch * world * max(seq_len - 1, 1)
    optimizer, scheduler, total_steps = build_optimizer_scheduler(
        model,
        train_cfg,
        tokens_per_step=tokens_per_step,
    )
    objective_type, epsilon, prefix_tables = resolve_objective(
        objective,
        tokenizer=tokenizer,
        device=device,
    )
    eval_cfg = config.get("eval") or {}
    lm_eval_cfg = eval_cfg.get("lm_eval") or {}
    tasks = list(lm_eval_cfg.get("tasks") or [])

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    tokens_seen = 0
    if rank == 0:
        write_metadata(output_dir, config, args.run_config)
    global_state, rank_state = load_checkpoint_state(
        checkpoint_dir,
        rank=rank,
        world=world,
    )
    if global_state:
        target = model.module if isinstance(model, DDP) else model
        target.load_state_dict(global_state["model"])
        optimizer.load_state_dict(global_state["optimizer"])
        if global_state.get("scheduler"):
            scheduler.load_state_dict(global_state["scheduler"])
        if global_state.get("dataloader"):
            loader.load_state_dict(global_state["dataloader"])
        start_step = int(global_state.get("step", 0))
        tokens_seen = int(global_state.get("tokens_seen", 0))
        LOGGER.info("resumed model/optimizer from %s at step %s", checkpoint_dir, start_step)
    if rank_state:
        if rank_state.get("rng"):
            restore_rng_state(rank_state["rng"])
        LOGGER.info("resumed dataloader state for rank %s", rank)

    if prefix_tables is not None:
        prefix_ids, prefix_weights_tensor, prefix_counts = prefix_tables
        prefix_tables = (
            prefix_ids.to(device),
            prefix_weights_tensor.to(device),
            prefix_counts.to(device),
        )

    model.train()
    it = iter(loader)
    step = start_step
    if tokens_seen == 0:
        tokens_seen = step * tokens_per_step

    def should_continue() -> bool:
        if max_steps > 0:
            return step < max_steps
        if max_tokens > 0:
            return tokens_seen < max_tokens
        return step < 1

    while should_continue():
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()

        with amp_ctx:
            logits = model(inputs).logits
            loss = compute_loss(
                logits,
                labels,
                objective_type=objective_type,
                epsilon=epsilon,
                prefix_tables=prefix_tables,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1
        tokens_seen += labels.numel() * world
        if rank == 0 and step % 10 == 0:
            LOGGER.info("step %s loss %.4f", step, float(loss))
        if checkpoint_enabled and save_every and step % save_every == 0:
            target = model.module if isinstance(model, DDP) else model
            save_checkpoint(
                checkpoint_dir,
                step,
                target,
                optimizer,
                scheduler,
                loader,
                tokens_seen,
                keep_last=keep_last,
                rank=rank,
                world=world,
            )

    if checkpoint_enabled:
        target = model.module if isinstance(model, DDP) else model
        save_checkpoint(
            checkpoint_dir,
            step,
            target,
            optimizer,
            scheduler,
            loader,
            tokens_seen,
            keep_last=keep_last,
            rank=rank,
            world=world,
        )

    if dist.is_initialized():
        dist.barrier()
    if rank == 0 and tasks:
        target = model.module if isinstance(model, DDP) else model
        target.eval()
        eval_dir = output_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        results = run_lm_eval(target, tokenizer, tasks, batch_size=1, device=device)
        out_path = eval_dir / "lm_eval_final.json"
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.info("lm-eval results saved to %s", out_path)
    if dist.is_initialized():
        dist.barrier()
        # Cleanly tear down DDP to avoid NCCL shutdown warnings.
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
