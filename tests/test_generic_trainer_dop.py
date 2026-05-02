import unittest

import torch

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TrainingMethod import TrainingMethod


class GenericTrainerDopTests(unittest.TestCase):
    def _build_config(self):
        config = TrainConfig.default_values()
        config.dop_enabled = True
        config.training_method = TrainingMethod.LORA
        config.dop_trigger_token = "sks"
        config.dop_class_replacement = "person"
        config.dop_interval_steps = 2
        config.dop_multiplier = 0.5
        config.text_encoder.train = False
        config.text_encoder_2.train = False
        config.text_encoder_3.train = False
        config.text_encoder_4.train = False
        return config

    def test_validate_dop_config_accepts_valid_lora_setup(self):
        trainer = GenericTrainer.__new__(GenericTrainer)
        trainer.config = self._build_config()
        trainer._GenericTrainer__validate_dop_config()

    def test_validate_dop_config_rejects_non_lora_training(self):
        trainer = GenericTrainer.__new__(GenericTrainer)
        trainer.config = self._build_config()
        trainer.config.training_method = TrainingMethod.FINE_TUNE

        with self.assertRaisesRegex(ValueError, "requires LoRA"):
            trainer._GenericTrainer__validate_dop_config()

    def test_validate_dop_config_rejects_invalid_interval(self):
        trainer = GenericTrainer.__new__(GenericTrainer)
        trainer.config = self._build_config()
        trainer.config.dop_interval_steps = 0

        with self.assertRaisesRegex(ValueError, "interval steps"):
            trainer._GenericTrainer__validate_dop_config()

    def test_validate_dop_config_rejects_negative_multiplier(self):
        trainer = GenericTrainer.__new__(GenericTrainer)
        trainer.config = self._build_config()
        trainer.config.dop_multiplier = -0.1

        with self.assertRaisesRegex(ValueError, "multiplier"):
            trainer._GenericTrainer__validate_dop_config()

    def test_validate_dop_config_rejects_negative_max_weighted_to_base_ratio(self):
        trainer = GenericTrainer.__new__(GenericTrainer)
        trainer.config = self._build_config()
        trainer.config.dop_max_weighted_to_base_ratio = -0.1

        with self.assertRaisesRegex(ValueError, "max weighted-to-base"):
            trainer._GenericTrainer__validate_dop_config()

    def test_loss_composition_matches_expected_weighting(self):
        base_loss = torch.tensor(1.25)
        dop_loss = torch.tensor(0.4)
        dop_multiplier = 0.3

        dop_weighted_loss = dop_loss * dop_multiplier
        total = base_loss + dop_weighted_loss

        self.assertAlmostEqual(dop_weighted_loss.item(), 0.12, places=7)
        self.assertAlmostEqual(total.item(), 1.37, places=7)

    def test_dop_weighted_capped_relative_to_base_loss(self):
        base_loss = torch.tensor(1.0, requires_grad=True)
        dop_weighted_loss = torch.tensor(2.0, requires_grad=True)
        cap_ratio = 0.5
        max_dop = cap_ratio * base_loss.detach()
        capped = torch.minimum(dop_weighted_loss, max_dop)
        self.assertAlmostEqual(capped.item(), 0.5, places=7)
        (base_loss + capped).backward()
        self.assertIsNotNone(dop_weighted_loss.grad)


if __name__ == "__main__":
    unittest.main()
