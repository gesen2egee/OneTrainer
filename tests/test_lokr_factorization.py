import unittest

import torch
from torch import nn

from modules.module.LoRAModule import LokrModule


class LokrFactorizationTests(unittest.TestCase):
    def test_initialize_weights_raises_for_invalid_explicit_factor(self):
        base = nn.Linear(13, 17, bias=False)
        with self.assertRaisesRegex(ValueError, "Invalid LoKr factorization"):
            LokrModule(
                prefix="transformer.invalid",
                orig_module=base,
                rank=4,
                alpha=1.0,
                factor=5,  # does not divide 13 or 17
            )

    def test_initialize_weights_succeeds_for_auto_factor(self):
        base = nn.Linear(13, 17, bias=False)
        module = LokrModule(
            prefix="transformer.auto",
            orig_module=base,
            rank=4,
            alpha=1.0,
            factor=-1,
        )
        self.assertTrue(
            module.lokr_w1 is not None or (module.lokr_w1_a is not None and module.lokr_w1_b is not None)
        )
        self.assertTrue(
            module.lokr_w2 is not None or (module.lokr_w2_a is not None and module.lokr_w2_b is not None)
        )


if __name__ == "__main__":
    unittest.main()
