import unittest

import torch

from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LossWeight import LossWeight


class _DummyFlowLossSetup(ModelSetupDiffusionLossMixin):
    def __init__(self):
        super().__init__()


class FlowMatchingLossWeightTests(unittest.TestCase):
    def _base_inputs(self):
        config = TrainConfig.default_values()
        config.batch_size = 1
        config.gradient_accumulation_steps = 1
        config.loss_weight_strength = 5.0
        batch = {"loss_weight": torch.ones((1,), dtype=torch.float32)}
        data = {
            "loss_type": "target",
            "predicted": torch.zeros((1, 4), dtype=torch.float32),
            "target": torch.ones((1, 4), dtype=torch.float32),
            "timestep": torch.tensor([0], dtype=torch.long),
        }
        return config, batch, data

    def test_flow_matching_supported_weight_modes_produce_finite_losses(self):
        setup = _DummyFlowLossSetup()
        config, batch, data = self._base_inputs()

        for weight_mode in (
            LossWeight.MIN_SNR_GAMMA,
            LossWeight.DEBIASED_ESTIMATION,
            LossWeight.P2,
        ):
            cfg = TrainConfig.default_values().from_dict(config.to_dict())
            cfg.loss_weight_fn = weight_mode
            out = setup._flow_matching_losses(
                batch=batch,
                data=data,
                config=cfg,
                train_device=torch.device("cpu"),
                sigmas=torch.linspace(0.1, 1.0, 10, dtype=torch.float32),
            )
            self.assertTrue(torch.isfinite(out).all(), f"non-finite loss for {weight_mode}")
            self.assertGreater(out.mean().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
