import unittest

import torch

from modules.modelLoader.Flux2ModelLoader import Flux2LoRALoader
from modules.modelSaver.flux2.Flux2LoRASaver import Flux2LoRASaver
from modules.util.convert.lora.convert_lora_util import convert_to_legacy_diffusers, convert_to_omi
from modules.util.convert.lora.convert_lora_util import convert_to_diffusers
from modules.util.convert.lora.convert_flux2_lora import convert_flux2_lora_key_sets


class Flux2LoraConversionTests(unittest.TestCase):
    def test_converts_flux2_diffusion_model_prefixes_to_transformer_prefixes(self):
        state_dict = {
            "diffusion_model.txt_in.lora_down.weight": torch.zeros((8, 4)),
            "diffusion_model.txt_in.lora_up.weight": torch.zeros((4, 8)),
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight": torch.zeros((8, 4)),
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_up.weight": torch.zeros((4, 8)),
            "diffusion_model.single_blocks.0.linear1.lora_down.weight": torch.zeros((8, 4)),
            "diffusion_model.single_blocks.0.linear1.lora_up.weight": torch.zeros((4, 8)),
            "diffusion_model.single_stream_modulation.lin.lora_down.weight": torch.zeros((8, 4)),
            "diffusion_model.single_stream_modulation.lin.lora_up.weight": torch.zeros((4, 8)),
        }

        converted = convert_to_diffusers(state_dict, convert_flux2_lora_key_sets())

        self.assertIn("transformer.context_embedder.lora_down.weight", converted)
        self.assertIn("transformer.context_embedder.lora_up.weight", converted)
        self.assertIn("transformer.transformer_blocks.0.img_attn.qkv.lora_down.weight", converted)
        self.assertIn("transformer.transformer_blocks.0.img_attn.qkv.lora_up.weight", converted)
        self.assertIn("transformer.single_transformer_blocks.0.linear1.lora_down.weight", converted)
        self.assertIn("transformer.single_transformer_blocks.0.linear1.lora_up.weight", converted)
        self.assertIn("transformer.single_stream_modulation.linear.lora_down.weight", converted)
        self.assertIn("transformer.single_stream_modulation.linear.lora_up.weight", converted)

    def test_flux2_loader_exposes_flux2_conversion_key_sets(self):
        loader = Flux2LoRALoader()
        key_sets = loader._get_convert_key_sets(model=None)

        self.assertIsNotNone(key_sets)
        self.assertTrue(any(k.omi_prefix == "diffusion_model.txt_in" for k in key_sets))

    def test_flux2_saver_uses_converter_for_omi_and_legacy_exports(self):
        saver = Flux2LoRASaver()
        key_sets = saver._get_convert_key_sets(model=None)
        self.assertIsNotNone(key_sets)

        state_dict = {
            "transformer.context_embedder.lora_down.weight": torch.zeros((8, 4)),
            "transformer.context_embedder.lora_up.weight": torch.zeros((4, 8)),
        }
        omi = convert_to_omi(state_dict, key_sets)
        legacy = convert_to_legacy_diffusers(state_dict, key_sets)

        self.assertIn("diffusion_model.txt_in.lora_down.weight", omi)
        self.assertIn("diffusion_model.txt_in.lora_up.weight", omi)
        self.assertIn("transformer_context_embedder.lora_down.weight", legacy)
        self.assertIn("transformer_context_embedder.lora_up.weight", legacy)

    def test_unknown_source_format_is_left_unchanged(self):
        state_dict = {
            "transformer.transformer_blocks.0.attn.to_q.lora_down.weight": torch.zeros((8, 4)),
            "transformer.transformer_blocks.0.attn.to_q.lora_up.weight": torch.zeros((4, 8)),
            "transformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(1.0),
        }

        converted = convert_to_omi(state_dict, convert_flux2_lora_key_sets())

        self.assertEqual(set(converted.keys()), set(state_dict.keys()))
        self.assertFalse(any("txt_intransformer" in k for k in converted))


if __name__ == "__main__":
    unittest.main()
