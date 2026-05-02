import unittest
from unittest import mock

import torch

from modules.util import sampler_only_lora


class _WrapperStub:
    def __init__(self, module_keys):
        self.lora_modules = {key: object() for key in module_keys}


class _SampleConfigStub:
    def __init__(self, source="sampler.safetensors", rank=None, strength=1.0):
        self.sampler_lora_model_name = source
        self.sampler_lora_rank = rank
        self.sampler_lora_strength = strength


class _TrainConfigStub:
    def __init__(self):
        self.model_type = "SD"
        self.lora_weight_dtype = "FLOAT32"


class SamplerOnlyLoRATests(unittest.TestCase):
    def test_maybe_download_hf_lora_resolve_url(self):
        url = "https://huggingface.co/org/repo/resolve/main/subdir/lora.safetensors"
        with mock.patch("modules.util.sampler_only_lora.huggingface_hub.hf_hub_download", return_value="C:/tmp/lora.safetensors") as download:
            path = sampler_only_lora._maybe_download_hf_lora(url)
        self.assertEqual(path, "C:/tmp/lora.safetensors")
        download.assert_called_once_with(
            repo_id="org/repo",
            filename="lora.safetensors",
            subfolder="subdir",
            revision="main",
        )

    def test_maybe_download_hf_lora_repo_id_tries_candidates(self):
        calls = []

        def fake_download(repo_id, filename, **kwargs):
            calls.append((repo_id, filename))
            if filename == "adapter_model.safetensors":
                return "C:/tmp/adapter_model.safetensors"
            raise RuntimeError("not found")

        with mock.patch("modules.util.sampler_only_lora.huggingface_hub.hf_hub_download", side_effect=fake_download):
            path = sampler_only_lora._maybe_download_hf_lora("org/repo")
        self.assertEqual(path, "C:/tmp/adapter_model.safetensors")
        self.assertEqual(
            calls,
            [
                ("org/repo", "pytorch_lora_weights.safetensors"),
                ("org/repo", "lora.safetensors"),
                ("org/repo", "adapter_model.safetensors"),
            ],
        )

    def test_maybe_download_hf_lora_invalid_hf_url_raises(self):
        with self.assertRaisesRegex(ValueError, "direct huggingface resolve/blob"):
            sampler_only_lora._maybe_download_hf_lora("https://huggingface.co/org/repo/tree/main")

    def test_infer_rank_from_lora_and_hada_keys(self):
        rank = sampler_only_lora.infer_lora_rank_from_state_dict(
            {"a.lora_down.weight": torch.zeros((8, 4))}
        )
        self.assertEqual(rank, 8)

        rank_hada = sampler_only_lora.infer_lora_rank_from_state_dict(
            {"a.hada_w1_a": torch.zeros((6, 3))}
        )
        self.assertEqual(rank_hada, 6)

    def test_remap_kohya_style_keys_for_wrapper(self):
        wrapper = _WrapperStub(["down.blocks.0.attn.to_q"])
        state_dict = {
            "lora_unet__down_blocks_0_attn_to_q.lora_down.weight": torch.ones((2, 2)),
            "unrelated.key": torch.ones((1,)),
        }
        remapped = sampler_only_lora._remap_kohya_style_state_keys_for_wrapper(
            state_dict,
            wrapper,
            "lora_unet",
        )
        self.assertIn("lora_unet.down.blocks.0.attn.to_q.lora_down.weight", remapped)
        self.assertIn("unrelated.key", remapped)

    def test_manager_reuses_and_tears_down_wrapper(self):
        manager = sampler_only_lora.SamplerOnlyLoRABatchManager()
        sample = _SampleConfigStub()
        train = _TrainConfigStub()
        model = object()
        device = torch.device("cpu")
        key = ("source", "mt", "prefix", 4, 1.0, "dtype", "cpu", "transformer:batch")
        fake_wrapper = mock.Mock()

        with mock.patch("modules.util.sampler_only_lora._create_sampler_wrapper", return_value=(fake_wrapper, ("d", "d", "d", 1, 1.0))) as create:
            manager.acquire(model, train, sample, device, key)
            manager.acquire(model, train, sample, device, key)
            self.assertEqual(create.call_count, 1)
            manager.close()
            fake_wrapper.remove_hook_from_module.assert_called_once()


if __name__ == "__main__":
    unittest.main()
