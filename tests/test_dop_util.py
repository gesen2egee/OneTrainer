import unittest

import torch

from modules.util import dop_util
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DOPPolicy import DOPPolicy


class _DummyTokenizer:
    model_max_length = 8

    def __call__(self, prompts, padding, truncation, return_tensors, max_length=None):
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        width = max_length or self.model_max_length
        rows = len(prompts)
        input_ids = torch.ones((rows, width), dtype=torch.long)
        attention_mask = torch.ones((rows, width), dtype=torch.long)
        return type("Tokenized", (), {"input_ids": input_ids, "attention_mask": attention_mask})


class _DummyModel:
    def __init__(self):
        self.tokenizer = _DummyTokenizer()
        self.tokenizer_1 = _DummyTokenizer()


class DopUtilTests(unittest.TestCase):
    def test_should_run_dop_periodic(self):
        config = TrainConfig.default_values()
        config.dop_enabled = True
        config.dop_policy = DOPPolicy.PERIODIC
        config.dop_interval_steps = 3
        self.assertTrue(dop_util.should_run_dop(config, 0))
        self.assertFalse(dop_util.should_run_dop(config, 1))
        self.assertFalse(dop_util.should_run_dop(config, 2))
        self.assertTrue(dop_util.should_run_dop(config, 3))

    def test_apply_preset_fast_sets_expected_values(self):
        config = TrainConfig.default_values()
        config.dop_adaptive_strength = 0.3
        dop_util.apply_preset(config, "fast")
        self.assertTrue(config.dop_enabled)
        self.assertEqual(config.dop_policy, DOPPolicy.ADAPTIVE)
        self.assertAlmostEqual(config.dop_multiplier, 0.8, places=7)
        self.assertEqual(config.dop_interval_steps, 8)
        self.assertAlmostEqual(config.dop_adaptive_strength, 1.0, places=7)

    def test_prompt_replacement_retokenizes(self):
        config = TrainConfig.default_values()
        config.dop_enabled = True
        config.dop_trigger_token = "sks"
        config.dop_class_replacement = "person"
        config.train_device = "cpu"

        model = _DummyModel()
        batch = {
            "prompt": ["a sks portrait", "landscape"],
            "tokens": torch.zeros((2, 8), dtype=torch.long),
            "tokens_mask": torch.zeros((2, 8), dtype=torch.long),
            "text_encoder_hidden_state": torch.zeros((2, 77, 768), dtype=torch.float32),
        }
        replaced = dop_util.create_prompt_replaced_batch(model, batch, config)
        self.assertIsNotNone(replaced)
        self.assertEqual(replaced["prompt"][0], "a person portrait")
        self.assertTrue(torch.equal(replaced["tokens"], torch.ones((2, 8), dtype=torch.long)))
        self.assertIsNone(replaced["text_encoder_hidden_state"])

    def test_count_dop_trigger_replaced_samples_single_caption(self):
        config = TrainConfig.default_values()
        config.dop_trigger_token = "sks"
        config.dop_class_replacement = "person"
        batch = {"prompt": ["a sks portrait", "no trigger here", "sks style"]}
        self.assertEqual(dop_util.count_dop_trigger_replaced_samples(batch, config), 2)

    def test_count_dop_trigger_replaced_samples_union_across_prompt_keys(self):
        config = TrainConfig.default_values()
        config.dop_trigger_token = "sks"
        config.dop_class_replacement = "person"
        batch = {
            "prompt": ["a sks portrait", "plain"],
            "prompt_2": ["sks only in second", "also sks"],
        }
        # index 0: replaced in prompt; index 1: replaced only in prompt_2
        self.assertEqual(dop_util.count_dop_trigger_replaced_samples(batch, config), 2)

    def test_count_dop_trigger_replaced_samples_same_index_both_keys(self):
        config = TrainConfig.default_values()
        config.dop_trigger_token = "sks"
        config.dop_class_replacement = "person"
        batch = {
            "prompt": ["a sks portrait"],
            "prompt_2": ["sks second line"],
        }
        self.assertEqual(dop_util.count_dop_trigger_replaced_samples(batch, config), 1)

    def test_dop_loss_zero_when_equal(self):
        a = torch.zeros((1, 4), dtype=torch.float32)
        b = torch.zeros((1, 4), dtype=torch.float32)
        loss = dop_util.dop_loss(a, b)
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_should_run_dop_honors_start_and_end_steps(self):
        config = TrainConfig.default_values()
        config.dop_enabled = True
        config.dop_policy = DOPPolicy.ALWAYS_ON
        config.dop_start_step = 2
        config.dop_end_step = 4

        self.assertFalse(dop_util.should_run_dop(config, 1))
        self.assertTrue(dop_util.should_run_dop(config, 2))
        self.assertTrue(dop_util.should_run_dop(config, 4))
        self.assertFalse(dop_util.should_run_dop(config, 5))

    def test_prompt_replacement_missing_trigger_returns_none_by_default(self):
        config = TrainConfig.default_values()
        config.dop_enabled = True
        config.dop_trigger_token = "sks"
        config.dop_class_replacement = "person"
        config.train_device = "cpu"
        config.dop_allow_missing_trigger = False

        model = _DummyModel()
        batch = {
            "prompt": ["a portrait", "landscape"],
            "tokens": torch.zeros((2, 8), dtype=torch.long),
            "tokens_mask": torch.zeros((2, 8), dtype=torch.long),
        }
        replaced = dop_util.create_prompt_replaced_batch(model, batch, config)
        self.assertIsNone(replaced)

    def test_prompt_replacement_missing_trigger_allowed(self):
        config = TrainConfig.default_values()
        config.dop_enabled = True
        config.dop_trigger_token = "sks"
        config.dop_class_replacement = "person"
        config.train_device = "cpu"
        config.dop_allow_missing_trigger = True

        model = _DummyModel()
        batch = {
            "prompt": ["a portrait", "landscape"],
            "tokens": torch.zeros((2, 8), dtype=torch.long),
            "tokens_mask": torch.zeros((2, 8), dtype=torch.long),
            "text_encoder_hidden_state": torch.zeros((2, 77, 768), dtype=torch.float32),
        }
        replaced = dop_util.create_prompt_replaced_batch(model, batch, config)
        self.assertIsNotNone(replaced)
        self.assertEqual(replaced["prompt"][0], "a portrait")
        self.assertTrue(torch.equal(replaced["tokens"], torch.ones((2, 8), dtype=torch.long)))
        self.assertIsNone(replaced["text_encoder_hidden_state"])

    def test_prompt_replacement_preserves_existing_token_device(self):
        config = TrainConfig.default_values()
        config.dop_enabled = True
        config.dop_trigger_token = "sks"
        config.dop_class_replacement = "person"
        # Simulate configured train device that may differ from runtime token/cache device.
        config.train_device = "cuda:1"

        model = _DummyModel()
        batch = {
            "prompt": ["a sks portrait"],
            "tokens": torch.zeros((1, 8), dtype=torch.long, device="cpu"),
            "tokens_mask": torch.zeros((1, 8), dtype=torch.long, device="cpu"),
        }
        replaced = dop_util.create_prompt_replaced_batch(model, batch, config)
        self.assertIsNotNone(replaced)
        self.assertEqual(str(replaced["tokens"].device), "cpu")
        self.assertEqual(str(replaced["tokens_mask"].device), "cpu")


if __name__ == "__main__":
    unittest.main()
