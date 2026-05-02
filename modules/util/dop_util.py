import copy
import re
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DOPPolicy import DOPPolicy


def should_run_dop(config: TrainConfig, global_step: int) -> bool:
    if not config.dop_enabled:
        return False

    if global_step < max(0, int(config.dop_start_step)):
        return False

    # Backward compatibility: older configs frequently persisted end_step=0 when users intended
    # "no end". Interpret this legacy shape as unbounded for always-on mode.
    legacy_no_end = (
        int(config.dop_end_step) == 0
        and int(config.dop_start_step) == 0
        and config.dop_policy == DOPPolicy.ALWAYS_ON
    )
    if not legacy_no_end and int(config.dop_end_step) >= 0 and global_step > int(config.dop_end_step):
        return False

    policy = config.dop_policy
    if policy == DOPPolicy.ALWAYS_ON:
        return True
    if policy == DOPPolicy.PERIODIC:
        interval = max(1, int(config.dop_interval_steps))
        return global_step % interval == 0
    if policy == DOPPolicy.ADAPTIVE:
        # Simple adaptive schedule: dense early, sparse later.
        strength = max(0.1, float(config.dop_adaptive_strength))
        early_interval = max(1, int(round(2 * strength)))
        late_interval = max(early_interval + 1, int(round(10 * strength)))
        pivot = max(100, int(round(500 * strength)))
        interval = early_interval if global_step < pivot else late_interval
        return global_step % interval == 0
    if policy == DOPPolicy.MANUAL:
        interval = max(1, int(config.dop_interval_steps))
        return global_step % interval == 0

    return False


def apply_preset(config: TrainConfig, preset: str):
    config.dop_enabled = True
    config.dop_start_step = 0
    config.dop_end_step = -1
    if preset == "quality":
        config.dop_policy = DOPPolicy.ALWAYS_ON
        config.dop_multiplier = 1.0
        config.dop_interval_steps = 1
        return
    if preset == "balanced":
        config.dop_policy = DOPPolicy.PERIODIC
        config.dop_multiplier = 1.0
        config.dop_interval_steps = 5
        return
    if preset == "fast":
        config.dop_policy = DOPPolicy.ADAPTIVE
        config.dop_multiplier = 0.8
        config.dop_interval_steps = 8
        config.dop_adaptive_strength = 1.0
        return


def dop_loss(predicted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted, reference, reduction="mean")


def count_dop_trigger_replaced_samples(batch: dict, config: TrainConfig) -> int:
    """
    Number of batch indices where at least one caption field (prompt / prompt_*) had the DOP trigger
    replaced by the class string (same rules as the DOP forward pass).
    """
    trigger = (config.dop_trigger_token or "").strip()
    replacement = (config.dop_class_replacement or "").strip()
    if not trigger or not replacement:
        return 0

    prompt_keys = [key for key in batch.keys() if key == "prompt" or key.startswith("prompt_")]
    if not prompt_keys:
        return 0

    replaced_indices: set[int] = set()
    for prompt_key in sorted(prompt_keys):
        original = batch.get(prompt_key)
        if original is None:
            continue
        orig_list = [str(p) for p in list(original)]
        replaced_list = _replace_prompt_sequence(
            orig_list,
            trigger=trigger,
            replacement=replacement,
            word_boundary_only=config.dop_word_boundary_only,
            case_sensitive=config.dop_case_sensitive,
        )
        if replaced_list is None:
            continue
        for i, (before, after) in enumerate(zip(orig_list, replaced_list, strict=False)):
            if before != after:
                replaced_indices.add(i)
    return len(replaced_indices)


def create_prompt_replaced_batch(model, batch: dict, config: TrainConfig) -> dict | None:
    trigger = (config.dop_trigger_token or "").strip()
    replacement = (config.dop_class_replacement or "").strip()
    if not trigger or not replacement:
        return None

    prompt_keys = [key for key in batch.keys() if key == "prompt" or key.startswith("prompt_")]
    if not prompt_keys:
        return None

    dop_batch = copy.copy(batch)

    changed_any = False
    for prompt_key in prompt_keys:
        original = batch[prompt_key]
        replaced_list = _replace_prompt_sequence(
            original,
            trigger=trigger,
            replacement=replacement,
            word_boundary_only=config.dop_word_boundary_only,
            case_sensitive=config.dop_case_sensitive,
        )
        if replaced_list is None:
            continue
        if replaced_list != list(original):
            changed_any = True
        dop_batch[prompt_key] = replaced_list
        _retokenize_prompt_key(model, batch, dop_batch, prompt_key, replaced_list, config.train_device)

    if not changed_any and not config.dop_allow_missing_trigger:
        return None

    # Force text re-encode for non-trained encoders by clearing cached hidden states.
    for key in list(dop_batch.keys()):
        if key.startswith("text_encoder_") and ("hidden_state" in key or "pooled_state" in key):
            dop_batch[key] = None
        if key == "text_encoder_hidden_state":
            dop_batch[key] = None

    return dop_batch


def _replace_prompt_sequence(
        prompts: Sequence[str],
        *,
        trigger: str,
        replacement: str,
        word_boundary_only: bool,
        case_sensitive: bool,
) -> list[str] | None:
    if prompts is None:
        return None
    flags = 0 if case_sensitive else re.IGNORECASE
    if word_boundary_only:
        pattern = re.compile(rf"\b{re.escape(trigger)}\b", flags=flags)
    else:
        pattern = re.compile(re.escape(trigger), flags=flags)

    return [pattern.sub(replacement, str(prompt)) for prompt in prompts]


def _retokenize_prompt_key(
        model,
        original_batch: dict,
        dop_batch: dict,
        prompt_key: str,
        prompts: list[str],
        train_device: str,
):
    suffix = prompt_key.removeprefix("prompt")
    token_key = f"tokens{suffix}" if suffix else "tokens"
    mask_key = f"tokens_mask{suffix}" if suffix else "tokens_mask"
    tokenizer_attr = f"tokenizer{suffix}" if suffix else "tokenizer"
    tokenizer = getattr(model, tokenizer_attr, None)
    if tokenizer is None:
        return

    max_length = None
    existing_tokens = original_batch.get(token_key)
    if existing_tokens is not None and hasattr(existing_tokens, "shape") and len(existing_tokens.shape) > 1:
        max_length = int(existing_tokens.shape[1])
    elif hasattr(tokenizer, "model_max_length"):
        max_length = int(tokenizer.model_max_length)

    kwargs = {
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        kwargs["max_length"] = max_length

    tokenized = tokenizer(prompts, **kwargs)

    target_device = None
    if existing_tokens is not None and hasattr(existing_tokens, "device"):
        target_device = existing_tokens.device
    elif mask_key in original_batch:
        existing_mask = original_batch.get(mask_key)
        if existing_mask is not None and hasattr(existing_mask, "device"):
            target_device = existing_mask.device

    if target_device is not None:
        dop_batch[token_key] = tokenized.input_ids.to(device=target_device)
        if mask_key in original_batch and getattr(tokenized, "attention_mask", None) is not None:
            dop_batch[mask_key] = tokenized.attention_mask.to(device=target_device)
    else:
        # Keep tokenizer output device unchanged (usually CPU) to avoid mismatches with offloaded encoders.
        dop_batch[token_key] = tokenized.input_ids
        if mask_key in original_batch and getattr(tokenized, "attention_mask", None) is not None:
            dop_batch[mask_key] = tokenized.attention_mask
