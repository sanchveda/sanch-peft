"""Causal-LM training loop with DDP + AMP, all stages visible inline.

Launch single-process:  python -m experiments.lora.scripts.train --config ...
Launch multi-GPU:       torchrun --standalone --nproc_per_node=N <same args>

The same code path covers both. Structure:
    train(...)        - one-time setup + cleanup
    _training_loop(...)    - the per-step body (LR + forward/backward + log + eval)

TrainConfig lives in core.training.config (typed YAML schema).
CausalLMDataset lives in core.data.base   (dataset contract).
evaluate(...) lives in core.training.evaluate (reused by validate.py).
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from core.data.base import CausalLMDataset
from core.training.config import TrainConfig
from core.training.evaluate import evaluate


def train(
    model: torch.nn.Module,
    dataset: CausalLMDataset,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> list[dict]:
    # ---- Stage 1: distributed setup (single-process if no torchrun env vars)
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, local_rank = 0, 0
        device = _auto_device(cfg.device)
    master = rank == 0

    # ---- Stage 2: precision (autocast ctx + GradScaler; both no-op when N/A)
    device_type = "cuda" if device.startswith("cuda") else device
    dtype = _resolve_dtype(device_type, cfg.dtype)
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        if device_type == "cuda" and dtype != "float32"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device_type == "cuda" and dtype == "float16")
    )

    # ---- Stage 3: model placement, seeding, DDP wrap, trainable-param check
    torch.manual_seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model.to(device)
    raw_model = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. Did you apply LoRA?")

    # ---- Stage 4 + 5: run the loop (training + periodic eval)
    history = _training_loop(
        model=model,
        raw_model=raw_model,
        dataset=dataset,
        optimizer=optimizer,
        trainable=trainable,
        cfg=cfg,
        ctx=ctx,
        scaler=scaler,
        device=device,
        master=master,
        ddp=ddp,
    )

    if ddp:
        dist.destroy_process_group()
    return history


def _training_loop(
    *,
    model: torch.nn.Module,
    raw_model: torch.nn.Module,
    dataset: CausalLMDataset,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    cfg: TrainConfig,
    ctx,
    scaler: torch.amp.GradScaler,
    device: str,
    master: bool,
    ddp: bool,
) -> list[dict]:
    """Per-step body: set LR, forward/backward with grad accum, step, log, maybe eval."""
    history: list[dict] = []
    x, y = dataset.get_batch("train", cfg.batch_size, cfg.block_size)
    x, y = x.to(device), y.to(device)

    for step in range(1, cfg.num_steps + 1):
        # ---- learning rate for this step
        lr = _get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ---- one optimizer step (forward/backward w/ grad accum, clip, step)
        for micro_step in range(cfg.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == cfg.gradient_accumulation_steps - 1
                )
            with ctx:
                _, loss = model(x, targets=y)
                loss = loss / cfg.gradient_accumulation_steps
            # prefetch next batch while backward runs on the GPU
            x, y = dataset.get_batch("train", cfg.batch_size, cfg.block_size)
            x, y = x.to(device), y.to(device)
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        train_loss = loss.item() * cfg.gradient_accumulation_steps
        # average the per-rank training loss for an accurate reported number.
        # gradients were already all-reduced by DDP backward; this is purely
        # for logging — one extra sync point per step.
        if ddp:
            loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = loss_tensor.item()

        record: dict = {"step": step, "lr": lr, "train_loss": train_loss}

        # ---- log
        if master and (step == 1 or step % cfg.log_interval == 0):
            print(f"step {step:>5} | lr {lr:.2e} | loss {train_loss:.4f}")

        # ---- periodic averaged eval (every rank evaluates, losses all-reduced)
        # all ranks MUST call evaluate() and the all_reduce — collectives hang
        # if some ranks skip them. Only master prints / would also be where
        # a checkpoint hook would live.
        if step % cfg.eval_interval == 0 or step == cfg.num_steps:
            metrics = evaluate(
                raw_model,
                dataset,
                cfg,
                splits=("train", "val"),
                ctx=ctx,
                device=device,
            )
            if ddp:
                for key, value in metrics.items():
                    value_tensor = torch.tensor([value], device=device)
                    dist.all_reduce(value_tensor, op=dist.ReduceOp.AVG)
                    metrics[key] = value_tensor.item()
            record.update(metrics)
            if master:
                print(
                    f"  eval | train {metrics['train_loss']:.4f} | "
                    f"val {metrics['val_loss']:.4f}"
                )

        history.append(record)

    return history


# ---- small helpers ---------------------------------------------------------


def _auto_device(want: str) -> str:
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device_type: str, dtype: str) -> str:
    if dtype != "auto":
        return dtype
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float32"


def _get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / (cfg.warmup_iters + 1)
    if step > cfg.num_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_iters) / max(1, cfg.num_steps - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)
