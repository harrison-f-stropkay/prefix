"""Dry-run resume check using the real training config and dataset."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from prefix.eval import run_eval_and_log
from prefix.logging_utils import configure_logging
from prefix.train import (
    build_dataloader,
    build_model_and_tokenizer,
    build_optimizer_scheduler,
    build_train_components,
    infer_run_dir,
    init_dist,
    load_checkpoint_state,
    restore_rng_state,
    run_train_step,
    save_checkpoint,
    select_eval_tasks,
    setup_run,
    unwrap_model,
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


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or infer_run_dir(args.run_config, Path("dry_runs"))
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir / "dry-run.log")
    rank, world, local_rank = init_dist()
    LOGGER.info("initialized dist (rank=%s world=%s local_rank=%s)", rank, world, local_rank)
    (
        config,
        device,
        amp_ctx,
        train_cfg,
        data_cfg,
        model_cfg,
        objective,
        eval_tasks,
        eval_limit,
        _eval_every,
        seed,
    ) = setup_run(args.run_config, local_rank=local_rank)
    fast_tasks = select_eval_tasks(eval_tasks, charbench_variant="fast")
    slow_tasks = select_eval_tasks(eval_tasks, charbench_variant="slow")

    per_device_batch = int(train_cfg.get("per_gpu_batch_size", 1))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    checkpoint_cfg = train_cfg.get("checkpointing") or {}
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", True))
    keep_last = int(checkpoint_cfg.get("keep_last", 1))
    shuffle = bool(train_cfg.get("shuffle", True))

    if not checkpoint_enabled:
        raise ValueError("Dry run requires checkpointing to be enabled.")

    (
        loader,
        model,
        tokenizer,
        optimizer,
        scheduler,
        objective_type,
        epsilon,
        prefix_tables,
        tokens_per_step,
    ) = build_train_components(
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        objective=objective,
        per_device_batch=per_device_batch,
        shuffle=shuffle,
        seed=seed,
        device=device,
        world=world,
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
    loss1, tokens = run_train_step(
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
    target = unwrap_model(model)
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
    if rank == 0 and fast_tasks:
        target_eval = unwrap_model(model)
        target_eval.eval()
        with torch.no_grad():
            run_eval_and_log(
                model=target_eval,
                tokenizer=tokenizer,
                tasks=fast_tasks,
                metrics_path=output_dir / "metrics.jsonl",
                step=step,
                tokens_seen=tokens_seen,
                eval_name="fast",
                label="dry-run fast lm-eval",
                batch_size=1,
                device=device,
                limit=None if eval_limit is None else int(eval_limit),
            )
        target_eval.train()

    batch2_before, it = next_batch(it, loader)
    loss2_before, tokens = run_train_step(
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
    resumed_optimizer, resumed_scheduler = build_optimizer_scheduler(
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
    target = unwrap_model(resumed_model)
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
    loss2_after, _ = run_train_step(
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
        LOGGER.info(
            "loss step1: %.6f; loss step2 before: %.6f; after: %.6f",
            loss1,
            loss2_before,
            loss2_after,
        )
        if slow_tasks:
            LOGGER.info("dry-run skipping slow eval; fast eval runs after step 1")
    # Cleanly tear down DDP to avoid NCCL shutdown warnings.
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
