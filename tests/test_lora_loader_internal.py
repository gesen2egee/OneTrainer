import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames
from modules.util.convert.lora.convert_flux2_lora import convert_flux2_lora_key_sets


class _TestInternalLoader(LoRALoaderMixin):
    def _get_convert_key_sets(self, model):
        return convert_flux2_lora_key_sets()

    def load(self, model, model_names: ModelNames):
        return self._load(model, model_names)


class LoRALoaderInternalTests(unittest.TestCase):
    def test_internal_loader_keeps_internal_keys_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir)
            (backup_dir / "meta.json").write_text("{}", encoding="utf-8")
            (backup_dir / "lora").mkdir()

            original_state = {
                # Intentionally not part of current Flux2 conversion keysets.
                "transformer.transformer_blocks.0.attn.to_q.lora_down.weight": torch.zeros((4, 4)),
                "transformer.transformer_blocks.0.attn.to_q.lora_up.weight": torch.zeros((4, 4)),
                "transformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(1.0),
            }
            save_file(original_state, str(backup_dir / "lora" / "lora.safetensors"))

            model = SimpleNamespace(lora_state_dict=None)
            loader = _TestInternalLoader()
            loader.load(model, ModelNames(lora=str(backup_dir)))

            self.assertIsNotNone(model.lora_state_dict)
            self.assertEqual(set(model.lora_state_dict.keys()), set(original_state.keys()))

    def test_internal_loader_repairs_malformed_transformer_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir)
            (backup_dir / "meta.json").write_text("{}", encoding="utf-8")
            (backup_dir / "lora").mkdir()

            malformed_state = {
                "diffusion_model.txt_intransformer.transformer_blocks.0.attn.to_q.lora_down.weight": torch.zeros((4, 4)),
                "diffusion_model.txt_intransformer.transformer_blocks.0.attn.to_q.lora_up.weight": torch.zeros((4, 4)),
                "diffusion_model.txt_intransformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(1.0),
            }
            save_file(malformed_state, str(backup_dir / "lora" / "lora.safetensors"))

            model = SimpleNamespace(lora_state_dict=None)
            loader = _TestInternalLoader()
            loader.load(model, ModelNames(lora=str(backup_dir)))

            self.assertIsNotNone(model.lora_state_dict)
            self.assertIn(
                "transformer.transformer_blocks.0.attn.to_q.lora_down.weight",
                model.lora_state_dict,
            )
            self.assertNotIn(
                "diffusion_model.txt_intransformer.transformer_blocks.0.attn.to_q.lora_down.weight",
                model.lora_state_dict,
            )


if __name__ == "__main__":
    unittest.main()
