import tempfile
import tkinter as tk
import unittest
from unittest import mock

from modules.util import dop_util
from modules.ui.TrainUI import TrainUI
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DOPPolicy import DOPPolicy
from modules.util.ui.UIState import UIState


class UIWiringSmokeTests(unittest.TestCase):
    def test_ui_state_parses_enum_name_and_legacy_string(self):
        root = tk.Tcl()
        cfg = TrainConfig.default_values()
        ui_state = UIState(root, cfg)
        dop_policy_var = ui_state.get_var("dop_policy")

        dop_policy_var.set("PERIODIC")
        self.assertEqual(cfg.dop_policy, DOPPolicy.PERIODIC)

        dop_policy_var.set("DOPPolicy.ALWAYS_ON")
        self.assertEqual(cfg.dop_policy, DOPPolicy.ALWAYS_ON)

    def test_dop_preset_apply_updates_config(self):
        cfg = TrainConfig.default_values()
        dop_util.apply_preset(cfg, "fast")

        self.assertEqual(cfg.dop_policy, DOPPolicy.ADAPTIVE)

    def test_train_ui_start_tensorboard_avoids_restart_when_running(self):
        ui = TrainUI.__new__(TrainUI)
        ui.train_config = TrainConfig.default_values()
        ui.train_config.workspace_dir = tempfile.gettempdir()
        ui.train_config.tensorboard_port = 6006
        ui.train_config.tensorboard_expose = False
        running_proc = mock.Mock()
        running_proc.poll.return_value = None
        ui.always_on_tensorboard_subprocess = running_proc
        ui._tensorboard_last_args = ["stale"]

        with mock.patch("modules.ui.TrainUI.subprocess.Popen") as popen:
            ui._start_always_on_tensorboard()
            popen.assert_not_called()

    def test_train_ui_toggle_always_on_calls_start_and_stop(self):
        ui = TrainUI.__new__(TrainUI)
        ui.training_thread = None
        ui.train_config = TrainConfig.default_values()
        ui._start_always_on_tensorboard = mock.Mock()
        ui._stop_always_on_tensorboard = mock.Mock()

        ui.train_config.tensorboard_always_on = True
        ui.train_config.tensorboard = True
        ui._on_always_on_tensorboard_toggle()
        ui._start_always_on_tensorboard.assert_called_once()

        ui.train_config.tensorboard_always_on = False
        ui._on_always_on_tensorboard_toggle()
        ui._stop_always_on_tensorboard.assert_called_once()


if __name__ == "__main__":
    unittest.main()
