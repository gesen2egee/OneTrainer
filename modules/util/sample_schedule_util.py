"""Helpers for few-step / distillation sampling (custom timesteps, DMD2-style schedules)."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers import SchedulerMixin


def parse_custom_diffusion_timesteps(raw: str) -> list[int] | None:
    """Parse comma- or semicolon-separated timestep indices. Empty / whitespace -> None."""
    if raw is None or not str(raw).strip():
        return None
    parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
    if not parts:
        return None
    return [int(p, 10) for p in parts]


def noise_scheduler_set_inference_timesteps(
        noise_scheduler: "SchedulerMixin",
        diffusion_steps: int,
        device: "torch.device",
        custom_diffusion_timesteps: str = "",
        force_last_timestep: bool = False,
):
    """
    Configure the scheduler for inference and return the timestep tensor to iterate.

    Matches prior behavior for force_last_timestep when no custom list is set.
    """
    import torch

    custom = parse_custom_diffusion_timesteps(custom_diffusion_timesteps)
    set_sig = inspect.signature(noise_scheduler.set_timesteps)
    supports_timesteps_kw = "timesteps" in set_sig.parameters

    if custom:
        if not supports_timesteps_kw:
            raise ValueError(
                "custom_diffusion_timesteps requires a scheduler that supports explicit timesteps "
                "(e.g. LCM or Euler). This scheduler does not accept timesteps= in set_timesteps()."
            )
        ts = torch.tensor(custom, device=device, dtype=torch.long)
        noise_scheduler.set_timesteps(num_inference_steps=len(custom), timesteps=ts, device=device)
        timesteps = noise_scheduler.timesteps
    else:
        noise_scheduler.set_timesteps(diffusion_steps, device=device)
        timesteps = noise_scheduler.timesteps

    if force_last_timestep:
        last_val = noise_scheduler.config.num_train_timesteps - 1
        last_timestep = torch.ones(1, device=device, dtype=torch.int64) * last_val
        first = timesteps[0]
        if int(first.item()) != int(last_val):
            if custom:
                merged = [last_val] + list(custom)
                ts = torch.tensor(merged, device=device, dtype=torch.long)
                noise_scheduler.set_timesteps(num_inference_steps=len(merged), timesteps=ts, device=device)
                timesteps = noise_scheduler.timesteps
            else:
                # Preserve original step count behavior: only prepend the terminal training step
                # if it's missing, rather than requesting extra scheduler steps.
                timesteps = torch.cat([last_timestep, timesteps])

    return timesteps
