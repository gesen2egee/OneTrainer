import unittest

import torch

from modules.util.sample_schedule_util import (
    noise_scheduler_set_inference_timesteps,
    parse_custom_diffusion_timesteps,
)


class _SchedulerWithTimesteps:
    def __init__(self):
        self.config = type("Cfg", (), {"num_train_timesteps": 1000})()
        self.timesteps = torch.tensor([9, 6, 3], dtype=torch.long)
        self.calls = []

    def set_timesteps(self, num_inference_steps, device=None, timesteps=None):
        self.calls.append((num_inference_steps, device, timesteps))
        if timesteps is not None:
            self.timesteps = timesteps
        else:
            self.timesteps = torch.arange(num_inference_steps - 1, -1, -1, dtype=torch.long, device=device)


class _SchedulerWithoutTimestepsKw:
    def __init__(self):
        self.config = type("Cfg", (), {"num_train_timesteps": 1000})()
        self.timesteps = torch.tensor([3, 2, 1], dtype=torch.long)

    def set_timesteps(self, diffusion_steps, device=None):
        self.timesteps = torch.arange(diffusion_steps - 1, -1, -1, dtype=torch.long, device=device)


class SampleScheduleUtilTests(unittest.TestCase):
    def test_parse_custom_diffusion_timesteps(self):
        self.assertIsNone(parse_custom_diffusion_timesteps(""))
        self.assertIsNone(parse_custom_diffusion_timesteps("   "))
        self.assertEqual(parse_custom_diffusion_timesteps("999, 749;499"), [999, 749, 499])
        with self.assertRaises(ValueError):
            parse_custom_diffusion_timesteps("999,abc")

    def test_custom_timesteps_are_applied_directly(self):
        scheduler = _SchedulerWithTimesteps()
        timesteps = noise_scheduler_set_inference_timesteps(
            scheduler,
            diffusion_steps=20,
            device=torch.device("cpu"),
            custom_diffusion_timesteps="999,749,499,249",
            force_last_timestep=False,
        )
        self.assertEqual(timesteps.tolist(), [999, 749, 499, 249])

    def test_force_last_timestep_prepends_last_train_step_for_custom_list(self):
        scheduler = _SchedulerWithTimesteps()
        timesteps = noise_scheduler_set_inference_timesteps(
            scheduler,
            diffusion_steps=20,
            device=torch.device("cpu"),
            custom_diffusion_timesteps="900,700",
            force_last_timestep=True,
        )
        self.assertEqual(timesteps.tolist(), [999, 900, 700])

    def test_custom_timesteps_require_scheduler_support(self):
        scheduler = _SchedulerWithoutTimestepsKw()
        with self.assertRaisesRegex(ValueError, "requires a scheduler that supports explicit timesteps"):
            noise_scheduler_set_inference_timesteps(
                scheduler,
                diffusion_steps=10,
                device=torch.device("cpu"),
                custom_diffusion_timesteps="999,499",
                force_last_timestep=False,
            )

    def test_force_last_timestep_prepends_once_for_non_custom_schedule(self):
        scheduler = _SchedulerWithTimesteps()
        timesteps = noise_scheduler_set_inference_timesteps(
            scheduler,
            diffusion_steps=4,
            device=torch.device("cpu"),
            custom_diffusion_timesteps="",
            force_last_timestep=True,
        )
        self.assertEqual(timesteps.tolist(), [999, 3, 2, 1, 0])
        # Ensure no extra scheduler-step inflation happened.
        self.assertEqual(len(timesteps), 5)

    def test_force_last_timestep_non_custom_is_noop_when_first_is_terminal(self):
        scheduler = _SchedulerWithTimesteps()
        scheduler.config.num_train_timesteps = 4
        timesteps = noise_scheduler_set_inference_timesteps(
            scheduler,
            diffusion_steps=4,
            device=torch.device("cpu"),
            custom_diffusion_timesteps="",
            force_last_timestep=True,
        )
        self.assertEqual(timesteps.tolist(), [3, 2, 1, 0])


if __name__ == "__main__":
    unittest.main()
