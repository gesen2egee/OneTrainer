import unittest

from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig


class SampleConfigBridgeTests(unittest.TestCase):
    def test_from_train_config_inherits_sampler_lora_defaults(self):
        train = TrainConfig.default_values()
        train.sampler_lora_model_name = "sampler.safetensors"
        train.sampler_lora_strength = 0.75
        train.sampler_lora_rank = 32

        sample = SampleConfig.default_values()
        sample.from_train_config(train)

        self.assertEqual(sample.sampler_lora_model_name, "sampler.safetensors")
        self.assertEqual(sample.sampler_lora_strength, 0.75)
        self.assertEqual(sample.sampler_lora_rank, 32)

    def test_from_train_config_keeps_sample_override_strength(self):
        train = TrainConfig.default_values()
        train.sampler_lora_strength = 0.65
        train.sampler_lora_rank = 24

        sample = SampleConfig.default_values()
        sample.sampler_lora_strength = 0.2
        sample.from_train_config(train)

        self.assertEqual(sample.sampler_lora_strength, 0.2)
        self.assertEqual(sample.sampler_lora_rank, 24)

    def test_from_train_config_replaces_default_strength_of_one(self):
        train = TrainConfig.default_values()
        train.sampler_lora_strength = 0.55

        sample = SampleConfig.default_values()
        sample.sampler_lora_strength = 1.0
        sample.from_train_config(train)

        self.assertEqual(sample.sampler_lora_strength, 0.55)


if __name__ == "__main__":
    unittest.main()
