"""Dry-run resume check using the real training config and dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from prefix.config import load_run_config
from prefix.train import (
    build_amp_context,
    build_dataloader,
    build_model_and_tokenizer,
    build_optimizer_scheduler,
    compute_loss,
    configure_logging,
    infer_run_dir,
    init_dist,
    load_checkpoint_state,
    resolve_objective,
    restore_rng_state,
    save_checkpoint,
    set_global_seed,
    write_metadata,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-config", required=True, type=Path)
    return parser.parse_args()


def next_batch(
    it: Iterator[dict[str, Any]],
    loader: Any,
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    try:
        batch = next(it)
        return batch, it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def run_step(
    batch: dict[str, Any],
    *,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    objective_type: str,
    epsilon: float,
    prefix_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    amp_ctx: Any,
    grad_clip: float,
) -> tuple[float, int]:
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
    return float(loss.detach().cpu()), labels.numel()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or infer_run_dir(args.run_config, Path("dry_runs"))
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir / "dry_run.log")
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
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    checkpoint_cfg = train_cfg.get("checkpointing") or {}
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", True))
    keep_last = int(checkpoint_cfg.get("keep_last", 1))
    shuffle = bool(train_cfg.get("shuffle", True))

    if not checkpoint_enabled:
        raise ValueError("Dry run requires checkpointing to be enabled.")

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

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_metadata(output_dir, config, args.run_config)

    model.train()
    it = iter(loader)
    step = 0
    tokens_seen = 0

    batch1, it = next_batch(it, loader)
    loss1, tokens = run_step(
        batch1,
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        objective_type=objective_type,
        epsilon=epsilon,
        prefix_tables=prefix_tables,
        amp_ctx=amp_ctx,
        grad_clip=grad_clip,
    )
    step += 1
    tokens_seen += tokens * world
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

    batch2_before, it = next_batch(it, loader)
    loss2_before, tokens = run_step(
        batch2_before,
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        objective_type=objective_type,
        epsilon=epsilon,
        prefix_tables=prefix_tables,
        amp_ctx=amp_ctx,
        grad_clip=grad_clip,
    )
    step += 1
    tokens_seen += tokens * world

    resumed_loader, _ = build_dataloader(
        data_cfg,
        per_device_batch=per_device_batch,
        shuffle=shuffle,
        seed=seed,
    )
    resumed_model, _ = build_model_and_tokenizer(model_cfg, data_cfg, device=device)
    resumed_optimizer, resumed_scheduler, _ = build_optimizer_scheduler(
        resumed_model,
        train_cfg,
        tokens_per_step=tokens_per_step,
    )
    global_state, rank_state = load_checkpoint_state(
        checkpoint_dir,
        rank=rank,
        world=world,
    )
    if not global_state or not rank_state:
        raise RuntimeError("Missing checkpoint state for dry-run resume.")
    target = resumed_model.module if isinstance(resumed_model, DDP) else resumed_model
    target.load_state_dict(global_state["model"])
    resumed_optimizer.load_state_dict(global_state["optimizer"])
    if global_state.get("scheduler"):
        resumed_scheduler.load_state_dict(global_state["scheduler"])
    if global_state.get("dataloader"):
        resumed_loader.load_state_dict(global_state["dataloader"])
    if rank_state.get("rng"):
        restore_rng_state(rank_state["rng"])

    resumed_model.train()
    resumed_it = iter(resumed_loader)
    batch2_after, resumed_it = next_batch(resumed_it, resumed_loader)
    loss2_after, _ = run_step(
        batch2_after,
        device=device,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        objective_type=objective_type,
        epsilon=epsilon,
        prefix_tables=prefix_tables,
        amp_ctx=amp_ctx,
        grad_clip=grad_clip,
    )

    before_ids = batch2_before["input_ids"].to(device)
    after_ids = batch2_after["input_ids"].to(device)
    if dist.is_initialized():
        gathered_before = [torch.empty_like(before_ids) for _ in range(world)]
        gathered_after = [torch.empty_like(after_ids) for _ in range(world)]
        dist.all_gather(gathered_before, before_ids)
        dist.all_gather(gathered_after, after_ids)
        global_before = [row for tensor in gathered_before for row in tensor.tolist()]
        global_after = [row for tensor in gathered_after for row in tensor.tolist()]
    else:
        global_before = before_ids.tolist()
        global_after = after_ids.tolist()

    if global_before != global_after:
        raise RuntimeError("Global batch 2 mismatch after resumption.")
    if not np.isclose(loss2_before, loss2_after, rtol=0.0, atol=0.0):
        raise RuntimeError("Step 2 loss mismatch after resumption.")

    if rank == 0:
        LOGGER.info(
            "dry-run resume OK; per-rank batch %s, world size %s, global batch %s",
            per_device_batch,
            world,
            per_device_batch * world,
        )
        LOGGER.info("loss step1: %.6f; loss step2 before: %.6f; after: %.6f", loss1, loss2_before, loss2_after)


if __name__ == "__main__":
    main()
