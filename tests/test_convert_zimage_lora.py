import unittest

import torch

from modules.modelLoader.ZImageModelLoader import ZImageLoRALoader
from modules.modelSaver.zImage.ZImageLoRASaver import ZImageLoRASaver
from modules.util.convert.lora.convert_lora_util import convert_to_diffusers
from modules.util.convert.lora.convert_zimage_lora import convert_zimage_lora_key_sets


class ZImageLoraConversionTests(unittest.TestCase):
    def test_converts_diffusers_and_kohya_style_prefixes_to_zimage_transformer(self):
        state_dict = {
            # diffusers style aliases
            "lora_transformer.transformer_blocks.0.attn.to_q.lora_down.weight": torch.zeros((8, 4)),
            "lora_transformer.transformer_blocks.0.attn.to_q.lora_up.weight": torch.zeros((4, 8)),
            # kohya/legacy style alias
            "lora_transformer_transformer_blocks_0_attn_to_q.lora_down.weight": torch.zeros((8, 4)),
        }

        converted = convert_to_diffusers(state_dict, convert_zimage_lora_key_sets())

        self.assertIn("transformer.transformer_blocks.0.attn.to_q.lora_down.weight", converted)
        self.assertIn("transformer.transformer_blocks.0.attn.to_q.lora_up.weight", converted)

    def test_zimage_loader_and_saver_use_zimage_converter(self):
        loader = ZImageLoRALoader()
        saver = ZImageLoRASaver()

        self.assertIsNotNone(loader._get_convert_key_sets(model=None))
        self.assertIsNotNone(saver._get_convert_key_sets(model=None))


if __name__ == "__main__":
    unittest.main()
