from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet


def convert_qwen_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    # Transformer aliases used by external checkpoints.
    keys += [LoraConversionKeySet("lora_transformer", "transformer")]
    keys += [LoraConversionKeySet("transformer", "transformer")]

    # Text-encoder aliases used by some Kohya-style exports.
    keys += [LoraConversionKeySet("lora_te", "text_encoder")]
    keys += [LoraConversionKeySet("lora_text_encoder", "text_encoder")]
    keys += [LoraConversionKeySet("text_encoder", "text_encoder")]

    return keys
