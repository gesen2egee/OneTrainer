from modules.model.BaseModel import BaseModel
from modules.model.QwenModel import QwenModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.convert.lora.convert_qwen_lora import convert_qwen_lora_key_sets
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.ModelNames import ModelNames


class QwenLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return convert_qwen_lora_key_sets()

    def load(
            self,
            model: QwenModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
