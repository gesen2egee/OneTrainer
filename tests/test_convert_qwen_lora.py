import unittest

import torch

from modules.modelLoader.qwen.QwenLoRALoader import QwenLoRALoader
from modules.modelSaver.qwen.QwenLoRASaver import QwenLoRASaver
from modules.util.convert.lora.convert_lora_util import convert_to_diffusers
from modules.util.convert.lora.convert_qwen_lora import convert_qwen_lora_key_sets


class QwenLoraConversionTests(unittest.TestCase):
    def test_converts_transformer_and_text_encoder_aliases(self):
        state_dict = {
            "lora_transformer.blocks.0.attn.to_q.lora_down.weight": torch.zeros((8, 4)),
            "lora_transformer.blocks.0.attn.to_q.lora_up.weight": torch.zeros((4, 8)),
            "lora_te.layers.0.self_attn.q_proj.lora_down.weight": torch.zeros((8, 4)),
        }

        converted = convert_to_diffusers(state_dict, convert_qwen_lora_key_sets())

        self.assertIn("transformer.blocks.0.attn.to_q.lora_down.weight", converted)
        self.assertIn("transformer.blocks.0.attn.to_q.lora_up.weight", converted)
        self.assertIn("text_encoder.layers.0.self_attn.q_proj.lora_down.weight", converted)

    def test_qwen_loader_and_saver_expose_conversion_key_sets(self):
        loader = QwenLoRALoader()
        saver = QwenLoRASaver()
        self.assertIsNotNone(loader._get_convert_key_sets(model=None))
        self.assertIsNotNone(saver._get_convert_key_sets(model=None))


if __name__ == "__main__":
    unittest.main()
