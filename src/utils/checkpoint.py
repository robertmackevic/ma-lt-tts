import logging
from pathlib import Path
from typing import Any, Mapping, Tuple, Optional, Dict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def latest_checkpoint_path(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    return paths[-1]


def load_checkpoint(
        path: Path,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        allow_partial_embeddings: bool = False,
) -> Tuple[float, int]:
    """
    Load a checkpoint:
      - Exact-shape matches are copied.
      - Mismatched tensors are skipped (or partially filled for embeddings if enabled).
      - Stats are logged (copied / partial / skipped / unexpected keys).
    """
    ckpt = torch.load(path, map_location="cpu")
    iteration = ckpt.get("iteration", 0)
    learning_rate = ckpt.get("learning_rate", 0.0)
    saved: Dict[str, torch.Tensor] = ckpt["model"]

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    mdl = model.module if hasattr(model, "module") else model
    cur = mdl.state_dict()

    new_state = {}
    stats = {
        "copied_tensors": 0,
        "copied_params": 0,
        "partial_tensors": 0,
        "partial_params": 0,
        "skipped_tensors": 0,
        "skipped_params": 0,
    }

    # Helper: optionally do safe row-wise copy for (usually) embeddings when only num_embeddings differs
    def maybe_partial_copy(dst: torch.Tensor, src: torch.Tensor) -> Optional[torch.Tensor]:
        if not allow_partial_embeddings:
            return None

        if dst.ndim == src.ndim == 2 and dst.shape[1] == src.shape[1]:
            rows = min(dst.shape[0], src.shape[0])
            out = dst.clone()
            out[:rows].copy_(src[:rows])
            return out

        return None

    for k, dst in cur.items():
        if k not in saved:
            # Missing in checkpoint: keep current init
            new_state[k] = dst
            stats["skipped_tensors"] += 1
            stats["skipped_params"] += dst.numel()
            logger.info("Skip (missing in ckpt): %s  shape=%s", k, tuple(dst.shape))
            continue

        src = saved[k]
        if src.shape == dst.shape and src.dtype == dst.dtype:
            new_state[k] = src
            stats["copied_tensors"] += 1
            stats["copied_params"] += src.numel()

        else:
            # Try partial (row-aligned) copy for enlarged embeddings/linear tables
            partial = maybe_partial_copy(dst, src)

            if partial is not None:
                new_state[k] = partial
                stats["partial_tensors"] += 1

                # Count only the params effectively copied
                stats["partial_params"] += min(dst.numel(), src.numel())
                logger.info(
                    "Partial copy: %s  ckpt=%s current=%s  (copied rows=%d)",
                    k, tuple(src.shape), tuple(dst.shape), min(dst.shape[0], src.shape[0])
                )

            else:
                # Shape or dtype mismatch: keep current init
                new_state[k] = dst
                stats["skipped_tensors"] += 1
                stats["skipped_params"] += dst.numel()
                logger.info(
                    "Skip (shape/dtype mismatch): %s  ckpt=%s current=%s, dtypes: ckpt=%s cur=%s",
                    k, tuple(src.shape), tuple(dst.shape), src.dtype, dst.dtype
                )

    # Keys present in checkpoint but not in the current model (informative)
    unexpected = sorted(set(saved.keys()) - set(cur.keys()))
    if unexpected:
        logger.info("Unexpected keys in checkpoint (%d): %s", len(unexpected), unexpected[:20])

    # Load the assembled state (all keys exist and have correct shapes now)
    mdl.load_state_dict(new_state, strict=True)

    logger.info(
        'Loaded checkpoint "%s" (iteration %d) | copied: %d tensors / %d params | '
        "partial: %d tensors / %d params | skipped: %d tensors / %d params",
        str(path),
        iteration,
        stats["copied_tensors"], stats["copied_params"],
        stats["partial_tensors"], stats["partial_params"],
        stats["skipped_tensors"], stats["skipped_params"],
    )

    return learning_rate, iteration


def save_checkpoint(model: nn.Module,
                    optimizer: Optimizer,
                    learning_rate: float,
                    iteration: int,
                    checkpoint_path: Path):
    logger.info('Saving model and optimizer state at iteration %d to %s', iteration, checkpoint_path)

    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    payload = {
        'model': state_dict,
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate
    }

    torch.save(payload, checkpoint_path)


def summarize(writer: SummaryWriter,
              global_step: int,
              scalars: Mapping[str, Any] = None,
              histograms: Mapping[str, Any] = None,
              images: Mapping[str, Any] = None,
              audios: Mapping[str, Any] = None,
              audio_sampling_rate: int = 22050):
    if scalars is None:
        scalars = {}
    if histograms is None:
        histograms = {}
    if images is None:
        images = {}
    if audios is None:
        audios = {}

    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)

    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)

    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')

    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)
