from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet


def convert_sana_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    # Accept external text-encoder LoRAs that use text_encoder.* and normalize to Sana's lora_te.*.
    keys += [LoraConversionKeySet("text_encoder", "lora_te")]

    # Accept external transformer LoRAs that use transformer.* and normalize to Sana's lora_transformer.*.
    keys += [LoraConversionKeySet("transformer", "lora_transformer")]

    # Keep already-normalized Sana keys stable across conversions.
    keys += [LoraConversionKeySet("lora_te", "lora_te")]
    keys += [LoraConversionKeySet("lora_transformer", "lora_transformer")]

    return keys
