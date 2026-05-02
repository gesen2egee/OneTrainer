import unittest

import torch

from modules.modelLoader.sana.SanaLoRALoader import SanaLoRALoader
from modules.modelSaver.sana.SanaLoRASaver import SanaLoRASaver
from modules.util.convert.lora.convert_lora_util import convert_to_diffusers
from modules.util.convert.lora.convert_sana_lora import convert_sana_lora_key_sets


class SanaLoraConversionTests(unittest.TestCase):
    def test_converts_diffusers_and_kohya_style_prefixes_to_sana_targets(self):
        state_dict = {
            # diffusers style examples
            "text_encoder.layers.0.self_attn.q_proj.lora_down.weight": torch.zeros((8, 4)),
            "text_encoder.layers.0.self_attn.q_proj.lora_up.weight": torch.zeros((4, 8)),
            "transformer.transformer_blocks.0.attn.to_q.lora_down.weight": torch.zeros((8, 4)),
            "transformer.transformer_blocks.0.attn.to_q.lora_up.weight": torch.zeros((4, 8)),
            # kohya/legacy style examples (underscore path)
            "lora_te_layers_0_self_attn_q_proj.lora_down.weight": torch.zeros((8, 4)),
            "lora_transformer_transformer_blocks_0_attn_to_q.lora_down.weight": torch.zeros((8, 4)),
        }

        converted = convert_to_diffusers(state_dict, convert_sana_lora_key_sets())

        self.assertIn("lora_te.layers.0.self_attn.q_proj.lora_down.weight", converted)
        self.assertIn("lora_te.layers.0.self_attn.q_proj.lora_up.weight", converted)
        self.assertIn("lora_transformer.transformer_blocks.0.attn.to_q.lora_down.weight", converted)
        self.assertIn("lora_transformer.transformer_blocks.0.attn.to_q.lora_up.weight", converted)

    def test_sana_loader_and_saver_use_sana_converter(self):
        loader = SanaLoRALoader()
        saver = SanaLoRASaver()

        self.assertIsNotNone(loader._get_convert_key_sets(model=None))
        self.assertIsNotNone(saver._get_convert_key_sets(model=None))


if __name__ == "__main__":
    unittest.main()
