"""Apply an extra LoRA on the main denoiser only during sampling, stacked on the training adapter."""

from __future__ import annotations

import os
import re
import traceback
from contextlib import AbstractContextManager
from copy import deepcopy
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import torch
import huggingface_hub
from safetensors.torch import load_file
from torch import nn

from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.modelLoader.Flux2ModelLoader import Flux2LoRALoader
from modules.modelLoader.ZImageModelLoader import ZImageLoRALoader
from modules.modelLoader.flux.FluxLoRALoader import FluxLoRALoader
from modules.modelLoader.qwen.QwenLoRALoader import QwenLoRALoader
from modules.modelLoader.stableDiffusion.StableDiffusionLoRALoader import StableDiffusionLoRALoader
from modules.modelLoader.stableDiffusionXL.StableDiffusionXLLoRALoader import StableDiffusionXLLoRALoader

if TYPE_CHECKING:
    from modules.model.BaseModel import BaseModel
    from modules.util.config.SampleConfig import SampleConfig
    from modules.util.config.TrainConfig import TrainConfig

HF_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REMAP_LOGGED: set[tuple[str, str]] = set()
_DIAG_LOGGED: set[tuple[str, str, str, int, float]] = set()
_REUSE_LOGGED: set[tuple[str, str, str, int, float, str, str, str]] = set()
_REBUILD_LOGGED: set[tuple[str, str, str, int, float, str, str, str]] = set()
_TEARDOWN_LOGGED: set[tuple[str, str, str, int, float, str, str, str]] = set()


def model_type_supports_sampler_only_lora(model_type: ModelType) -> bool:
    return (
            model_type.is_stable_diffusion_xl()
            or model_type.is_stable_diffusion()
            or model_type.is_z_image()
            or model_type.is_flux_1()
            or model_type.is_flux_2()
            or model_type.is_qwen()
    )


def _sampler_lora_target(model: BaseModel, model_type: ModelType) -> tuple[nn.Module, str] | None:
    """
    Return (parent_module, state_dict_prefix) for the sampler-only adapter.
    Prefixes must match *LoRASetup (e.g. ZImageLoRASetup, FluxLoRASetup).
    """
    if model_type.is_stable_diffusion_xl() or model_type.is_stable_diffusion():
        unet = getattr(model, "unet", None)
        if unet is None:
            return None
        return unet, "lora_unet"
    if model_type.is_z_image():
        tr = getattr(model, "transformer", None)
        if tr is None:
            return None
        return tr, "transformer"
    if model_type.is_flux_1():
        tr = getattr(model, "transformer", None)
        if tr is None:
            return None
        return tr, "lora_transformer"
    if model_type.is_flux_2():
        tr = getattr(model, "transformer", None)
        if tr is None:
            return None
        return tr, "transformer"
    if model_type.is_qwen():
        tr = getattr(model, "transformer", None)
        if tr is None:
            return None
        return tr, "transformer"
    return None


def infer_lora_rank_from_state_dict(state_dict: dict) -> int:
    for k, v in state_dict.items():
        if k.endswith(".lora_down.weight"):
            return int(v.shape[0])
        if k.endswith(".hada_w1_a"):
            return int(v.shape[0])
    raise ValueError("Could not infer LoRA rank (no .lora_down.weight / .hada_w1_a tensors in state dict)")


def _load_raw_lora_file(path: str) -> dict:
    path = path.strip()
    if path.endswith(".ckpt"):
        return torch.load(path, weights_only=True)
    return load_file(path)


def _maybe_download_hf_lora(path: str) -> str:
    trimmed = path.strip()
    parsed = urlparse(trimmed)
    if parsed.scheme == "https" and parsed.netloc in {"huggingface.co", "huggingface.com"}:
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) >= 5 and parts[2] in {"resolve", "blob"}:
            repo_id = f"{parts[0]}/{parts[1]}"
            revision = parts[3]
            filename_rel = "/".join(parts[4:])
            subfolder, filename = os.path.split(filename_rel)
            return huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder if subfolder else None,
                revision=revision,
            )
        raise ValueError(
            "Sampler-only LoRA URL must be a direct huggingface resolve/blob file URL."
        )

    if HF_REPO_RE.match(trimmed):
        for candidate in (
                "pytorch_lora_weights.safetensors",
                "lora.safetensors",
                "adapter_model.safetensors",
        ):
            try:
                return huggingface_hub.hf_hub_download(repo_id=trimmed, filename=candidate)
            except Exception:
                continue
        raise ValueError(
            f"Could not infer LoRA filename for HF repo '{trimmed}'. "
            "Provide a direct file URL or a local file path."
        )

    return trimmed


def load_converted_sampler_lora_state_dict(model: BaseModel, path: str, model_type: ModelType) -> dict:
    if model_type.is_stable_diffusion_xl():
        lora_loader = StableDiffusionXLLoRALoader()
    elif model_type.is_stable_diffusion():
        lora_loader = StableDiffusionLoRALoader()
    elif model_type.is_z_image():
        lora_loader = ZImageLoRALoader()
    elif model_type.is_flux_1():
        lora_loader = FluxLoRALoader()
    elif model_type.is_flux_2():
        lora_loader = Flux2LoRALoader()
    elif model_type.is_qwen():
        lora_loader = QwenLoRALoader()
    else:
        raise RuntimeError(f"No sampler-only LoRA conversion loader for {model_type}")

    state_dict = _load_raw_lora_file(path)
    key_sets = lora_loader._get_convert_key_sets(model)
    if key_sets is not None:
        from modules.util.convert.lora.convert_lora_util import convert_to_diffusers

        state_dict = convert_to_diffusers(state_dict, key_sets)
    return state_dict


def _apply_sampler_lora_strength(wrapper: LoRAModuleWrapper, strength: float) -> None:
    if strength == 1.0:
        return
    for module in wrapper.lora_modules.values():
        if hasattr(module, "alpha") and getattr(module, "alpha", None) is not None:
            module.alpha.mul_(strength)


def _count_prefixed_state_keys(state_dict: dict, prefix: str) -> int:
    pref = prefix + "."
    return sum(1 for k in state_dict.keys() if k.startswith(pref))


def _remap_kohya_style_state_keys_for_wrapper(
        state_dict: dict[str, torch.Tensor],
        wrapper: LoRAModuleWrapper,
        target_prefix: str,
) -> dict[str, torch.Tensor]:
    """
    Remap keys like `lora_unet__a_b_c.lora_down.weight` to
    `<target_prefix>.a.b.c.lora_down.weight` by matching against actual wrapper
    module names (`wrapper.lora_modules.keys()`).
    """
    module_keys = list(wrapper.lora_modules.keys())
    underscore_lookup = {k.replace(".", "_"): k for k in module_keys}

    remapped: dict[str, torch.Tensor] = {}
    converted = 0
    for key, value in state_dict.items():
        if "__" not in key:
            remapped[key] = value
            continue

        left, right = key.split("__", 1)
        if not left.startswith("lora_"):
            remapped[key] = value
            continue
        if "." not in right:
            remapped[key] = value
            continue

        module_token, suffix = right.split(".", 1)
        module_name = underscore_lookup.get(module_token)
        if module_name is None:
            remapped[key] = value
            continue

        new_key = f"{target_prefix}.{module_name}.{suffix}"
        remapped[new_key] = value
        converted += 1

    if converted > 0:
        key = (target_prefix, str(converted))
        if key not in _REMAP_LOGGED:
            _REMAP_LOGGED.add(key)
            print(f"Sampler-only LoRA: remapped {converted} Kohya-style keys to '{target_prefix}.*'")
    return remapped


def build_sampler_lora_reuse_key(
        train_config: TrainConfig,
        sample_config: SampleConfig,
        train_device: torch.device,
        batch_marker: str = "",
) -> tuple[str, str, str, int, float, str, str, str]:
    source = (sample_config.sampler_lora_model_name or "").strip()
    model_type = str(train_config.model_type)
    # Derive stable target prefix from model type.
    if train_config.model_type.is_stable_diffusion_xl() or train_config.model_type.is_stable_diffusion():
        target_prefix = "lora_unet"
        target_kind = "unet"
    elif train_config.model_type.is_flux_1():
        target_prefix = "lora_transformer"
        target_kind = "transformer"
    else:
        target_prefix = "transformer"
        target_kind = "transformer"

    rank_val = sample_config.sampler_lora_rank if sample_config.sampler_lora_rank is not None else -1
    strength = float(sample_config.sampler_lora_strength)
    dtype = str(train_config.lora_weight_dtype)
    device = str(train_device)
    return (source, model_type, target_prefix, int(rank_val), strength, dtype, device, f"{target_kind}:{batch_marker}")


def _create_sampler_wrapper(
        model: BaseModel,
        train_config: TrainConfig,
        sample_config: SampleConfig,
        train_device: torch.device,
) -> tuple[LoRAModuleWrapper | None, tuple[str, str, str, int, float] | None]:
    source = (sample_config.sampler_lora_model_name or "").strip()
    if not source:
        return None, None
    try:
        path = _maybe_download_hf_lora(source)
    except Exception:
        traceback.print_exc()
        print("Sampler-only LoRA: failed to resolve HF link/repo; skipping.")
        return None, None
    if not os.path.isfile(path):
        print(f"Sampler-only LoRA: file not found: {source}")
        return None, None
    if not model_type_supports_sampler_only_lora(train_config.model_type):
        print("Sampler-only LoRA: unsupported model type, skipping.")
        return None, None
    if train_config.training_method == TrainingMethod.LORA:
        if train_config.peft_type != PeftType.LORA:
            print("Sampler-only LoRA: training uses a non-LoRA adapter type; skipping sampler LoRA.")
            return None, None
    elif train_config.training_method != TrainingMethod.FINE_TUNE:
        print("Sampler-only LoRA: only supported for LoRA or fine-tune training; skipping.")
        return None, None
    target = _sampler_lora_target(model, train_config.model_type)
    if target is None:
        print("Sampler-only LoRA: no denoiser module (unet/transformer) on model, skipping.")
        return None, None
    parent, prefix = target

    raw_state_dict = load_converted_sampler_lora_state_dict(
        model, path, train_config.model_type,
    )
    prefixed_keys = _count_prefixed_state_keys(raw_state_dict, prefix)
    rank = sample_config.sampler_lora_rank
    if rank is None:
        rank = infer_lora_rank_from_state_dict(raw_state_dict)
    cfg = deepcopy(train_config)
    cfg.lora_rank = rank
    layer_filter = []
    sampler_wrapper = LoRAModuleWrapper(
        parent, prefix, cfg, layer_filter,
    )
    state_dict = raw_state_dict
    if prefixed_keys == 0:
        state_dict = _remap_kohya_style_state_keys_for_wrapper(
            raw_state_dict, sampler_wrapper, prefix,
        )
        prefixed_keys = _count_prefixed_state_keys(state_dict, prefix)
    created_module_count = len(sampler_wrapper.lora_modules)
    sampler_wrapper.load_state_dict(state_dict, strict=False)
    sampler_wrapper.prune()
    loaded_module_count = len(sampler_wrapper.lora_modules)
    diag_key = (
        source,
        str(train_config.model_type),
        prefix,
        int(rank),
        float(sample_config.sampler_lora_strength),
    )
    if diag_key not in _DIAG_LOGGED:
        _DIAG_LOGGED.add(diag_key)
        print(
            "Sampler-only LoRA diagnostics: "
            f"source='{source}', resolved='{path}', model_type={train_config.model_type}, "
            f"prefix='{prefix}', state_keys={len(state_dict)}, prefixed_keys={prefixed_keys}, "
            f"created_modules={created_module_count}, loaded_modules={loaded_module_count}, "
            f"rank={rank}, strength={float(sample_config.sampler_lora_strength)}"
        )
    if loaded_module_count == 0:
        print(
            "Sampler-only LoRA: zero modules loaded for target prefix. "
            "Skipping hook. Check that this LoRA matches the selected model family/base."
        )
        return None, None
    sampler_wrapper.set_dropout(0.0)
    sampler_wrapper.to(
        train_device,
        dtype=train_config.lora_weight_dtype.torch_dtype(),
    )
    sampler_wrapper.requires_grad_(False)
    _apply_sampler_lora_strength(
        sampler_wrapper, float(sample_config.sampler_lora_strength),
    )
    sampler_wrapper.hook_to_module()
    return sampler_wrapper, diag_key


class SamplerOnlyLoRABatchManager:
    def __init__(self):
        self._active_key: tuple[str, str, str, int, float, str, str, str] | None = None
        self._active_diag_key: tuple[str, str, str, int, float] | None = None
        self._active_wrapper: LoRAModuleWrapper | None = None

    def acquire(
            self,
            model: BaseModel,
            train_config: TrainConfig,
            sample_config: SampleConfig,
            train_device: torch.device,
            batch_key: tuple[str, str, str, int, float, str, str, str] | None = None,
    ):
        key = batch_key or build_sampler_lora_reuse_key(train_config, sample_config, train_device)
        source = (sample_config.sampler_lora_model_name or "").strip()
        if not source:
            self.close()
            return

        if self._active_wrapper is not None and self._active_key == key:
            if key not in _REUSE_LOGGED:
                _REUSE_LOGGED.add(key)
                print("Sampler-only LoRA manager: reuse hit")
            return

        if self._active_wrapper is not None and self._active_key != key:
            if key not in _REBUILD_LOGGED:
                _REBUILD_LOGGED.add(key)
                print("Sampler-only LoRA manager: rebuild due to key change")
            self.close()

        wrapper, diag_key = _create_sampler_wrapper(model, train_config, sample_config, train_device)
        self._active_wrapper = wrapper
        self._active_diag_key = diag_key
        self._active_key = key if wrapper is not None else None

    def release(self):
        return

    def close(self):
        if self._active_wrapper is not None:
            self._active_wrapper.remove_hook_from_module()
            if self._active_key is not None and self._active_key not in _TEARDOWN_LOGGED:
                _TEARDOWN_LOGGED.add(self._active_key)
                print("Sampler-only LoRA manager: teardown")
        self._active_wrapper = None
        self._active_key = None
        self._active_diag_key = None


class SamplerOnlyLoRAContext(AbstractContextManager):
    """
    Builds a second LoRAModuleWrapper on the main denoiser (UNet or transformer) and hooks it
    after the training adapter so forwards compose as base + train_delta + sampler_delta.
    Removed in ``__exit__`` (LIFO unhook).
    """

    def __init__(
            self,
            model: BaseModel,
            train_config: TrainConfig,
            sample_config: SampleConfig,
            train_device: torch.device,
    ):
        self.model = model
        self.train_config = train_config
        self.sample_config = sample_config
        self.train_device = train_device
        self._manager = SamplerOnlyLoRABatchManager()

    def __enter__(self) -> SamplerOnlyLoRAContext:
        self._manager.acquire(
            self.model,
            self.train_config,
            self.sample_config,
            self.train_device,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._manager.close()
        return None
