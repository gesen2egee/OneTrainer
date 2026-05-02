import unittest
from enum import Enum

from modules.util.config.BaseConfig import BaseConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DOPPolicy import DOPPolicy


class _DemoEnum(Enum):
    MODE_A = "mode-a"
    MODE_B = "mode-b"

    def __str__(self):
        return self.value


class _DemoConfig(BaseConfig):
    @staticmethod
    def default_values():
        return _DemoConfig([("mode", _DemoEnum.MODE_A, _DemoEnum, False)])


class ConfigEnumParsingTests(unittest.TestCase):
    def test_base_config_parses_enum_member_name(self):
        cfg = _DemoConfig.default_values().from_dict({"mode": "MODE_B"})
        self.assertEqual(cfg.mode, _DemoEnum.MODE_B)

    def test_base_config_parses_legacy_enum_member_string(self):
        cfg = _DemoConfig.default_values().from_dict({"mode": "_DemoEnum.MODE_B"})
        self.assertEqual(cfg.mode, _DemoEnum.MODE_B)

    def test_base_config_parses_enum_value_string(self):
        cfg = _DemoConfig.default_values().from_dict({"mode": "mode-b"})
        self.assertEqual(cfg.mode, _DemoEnum.MODE_B)

    def test_train_config_parses_dop_policy_from_legacy_string(self):
        config = TrainConfig.default_values()
        payload = config.to_dict()
        payload["dop_policy"] = "DOPPolicy.PERIODIC"
        payload["dop_enabled"] = True
        payload["dop_trigger_token"] = "sks"
        payload["dop_class_replacement"] = "person"
        payload["sampler_lora_model_name"] = "sampler.safetensors"

        restored = TrainConfig.default_values().from_dict(payload)
        self.assertEqual(restored.dop_policy, DOPPolicy.PERIODIC)
        self.assertTrue(restored.dop_enabled)
        self.assertEqual(restored.sampler_lora_model_name, "sampler.safetensors")


if __name__ == "__main__":
    unittest.main()
