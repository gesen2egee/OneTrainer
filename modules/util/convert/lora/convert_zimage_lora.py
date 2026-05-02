from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet


def convert_zimage_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    # Normalize external lora_transformer.* style keys into ZImage's transformer.* namespace.
    keys += [LoraConversionKeySet("lora_transformer", "transformer")]

    # Keep already-normalized keys stable.
    keys += [LoraConversionKeySet("transformer", "transformer")]

    return keys
