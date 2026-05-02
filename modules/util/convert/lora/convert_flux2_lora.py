from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, map_prefix_range


def __map_double_transformer_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("img_attn.qkv", "img_attn.qkv", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_attn.qkv", "txt_attn.qkv", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_attn.proj", "img_attn.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_attn.proj", "txt_attn.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_mlp.0", "img_mlp.0", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_mlp.2", "img_mlp.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_mlp.0", "txt_mlp.0", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_mlp.2", "txt_mlp.2", parent=key_prefix)]

    return keys


def __map_single_transformer_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("linear1", "linear1", parent=key_prefix)]
    keys += [LoraConversionKeySet("linear2", "linear2", parent=key_prefix)]

    return keys


def convert_flux2_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []
    key_prefix = LoraConversionKeySet("diffusion_model", "transformer")

    keys += [LoraConversionKeySet("txt_in", "context_embedder", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_in", "x_embedder", parent=key_prefix)]
    keys += [LoraConversionKeySet("time_in.in_layer", "time_guidance_embed.timestep_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("time_in.out_layer", "time_guidance_embed.timestep_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("final_layer.linear", "proj_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("final_layer.adaLN_modulation.1", "norm_out.linear", parent=key_prefix, swap_chunks=True)]
    keys += [LoraConversionKeySet("double_stream_modulation_img.lin", "double_stream_modulation_img.linear", parent=key_prefix)]
    keys += [LoraConversionKeySet("double_stream_modulation_txt.lin", "double_stream_modulation_txt.linear", parent=key_prefix)]
    keys += [LoraConversionKeySet("single_stream_modulation.lin", "single_stream_modulation.linear", parent=key_prefix)]

    for k in map_prefix_range("double_blocks", "transformer_blocks", parent=key_prefix):
        keys += __map_double_transformer_block(k)

    for k in map_prefix_range("single_blocks", "single_transformer_blocks", parent=key_prefix):
        keys += __map_single_transformer_block(k)

    return keys
