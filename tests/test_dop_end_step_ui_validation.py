import unittest
import tkinter as tk

from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.UIState import UIState
from modules.util.ui.validation import FieldValidator
from modules.util.ui.validation_helpers import check_non_negative_or_minus_one


class DopEndStepUIValidationTests(unittest.TestCase):
    def test_dop_end_step_allows_minus_one(self):
        root = tk.Tcl()
        cfg = TrainConfig.default_values()
        ui_state = UIState(root, cfg)

        # We only need FieldValidator.validate()'s pure validation logic.
        # Avoid constructing the full widget-based FieldValidator (it needs a real Tk widget master).
        validator = FieldValidator.__new__(FieldValidator)
        validator.ui_state = ui_state
        validator.var_name = "dop_end_step"
        validator._required = False
        validator._extra_validate = check_non_negative_or_minus_one()

        self.assertIsNone(validator.validate("-1"))
        self.assertEqual(
            validator.validate("-2"),
            "Value must be -1 or a non-negative integer",
        )


if __name__ == "__main__":
    unittest.main()

