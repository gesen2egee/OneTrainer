import os
import traceback
from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, convert_to_diffusers
from modules.util.ModelNames import ModelNames

import torch

from safetensors.torch import load_file


class LoRALoaderMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        pass

    def __load_safetensors(
            self,
            model: BaseModel,
            lora_name: str,
    ):
        state_dict = load_file(lora_name)

        key_sets = self._get_convert_key_sets(model)
        if key_sets is not None:
            state_dict = convert_to_diffusers(state_dict, key_sets)

        model.lora_state_dict = state_dict

    def __load_ckpt(
            self,
            model: BaseModel,
            lora_name: str,
    ):
        state_dict = torch.load(lora_name, weights_only=True)

        key_sets = self._get_convert_key_sets(model)
        if key_sets is not None:
            state_dict = convert_to_diffusers(state_dict, key_sets)

        model.lora_state_dict = state_dict

    def __load_internal(
            self,
            model: BaseModel,
            lora_name: str,
    ):
        if os.path.exists(os.path.join(lora_name, "meta.json")):
            safetensors_lora_name = os.path.join(lora_name, "lora", "lora.safetensors")
            if os.path.exists(safetensors_lora_name):
                # Internal backups are already written in OneTrainer-native keyspace.
                # Re-running external format conversion here can corrupt unknown/legacy
                # keys and break resume loading for older checkpoints.
                model.lora_state_dict = self.__repair_internal_keys(load_file(safetensors_lora_name))
        else:
            raise Exception("not an internal model")

    @staticmethod
    def __repair_internal_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        repaired: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            fixed_key = key
            # Recovery path for malformed Flux2 keys like:
            # "diffusion_model.txt_intransformer.transformer_blocks..."
            # which should be "transformer.transformer_blocks..."
            if "transformer." in fixed_key and not fixed_key.startswith("transformer."):
                fixed_key = fixed_key[fixed_key.find("transformer."):]
            repaired[fixed_key] = value
        return repaired

    def _load(
            self,
            model: BaseModel,
            model_names: ModelNames,
    ):
        stacktraces = []

        if model_names.lora == "":
            return

        try:
            self.__load_internal(model, model_names.lora)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        if model_names.lora.endswith(".ckpt"):
            try:
                self.__load_ckpt(model, model_names.lora)
                return
            except Exception:
                stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(model, model_names.lora)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load LoRA: " + model_names.lora)
