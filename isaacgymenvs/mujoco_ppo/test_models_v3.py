import unittest

import torch

from mujoco_ppo.models_v3 import (
    AsymmetricActorCritic,
    AsymmetricModelConfig,
    CensoredNormal,
)


class CensoredNormalTest(unittest.TestCase):
    def test_samples_and_deterministic_mean_respect_bounds(self):
        torch.manual_seed(7)
        loc = torch.tensor([[0.0, 1.2, -1.3]])
        scale = torch.tensor([[0.2, 0.3, 0.4]])
        dist = CensoredNormal(loc, scale, action_clip=1.0)

        samples = dist.sample((4096,))

        self.assertTrue(torch.all(samples.abs() <= 1.0))
        self.assertTrue(torch.equal(dist.mean, torch.clamp(loc, -1.0, 1.0)))

    def test_boundary_log_prob_uses_gaussian_tail_mass(self):
        loc = torch.tensor([[0.0, 1.2, -1.3]])
        scale = torch.tensor([[0.2, 0.3, 0.4]])
        dist = CensoredNormal(loc, scale, action_clip=1.0)

        low = torch.full_like(loc, -1.0)
        high = torch.full_like(loc, 1.0)
        expected_low = torch.special.log_ndtr((-1.0 - loc) / scale)
        expected_high = torch.special.log_ndtr((loc - 1.0) / scale)

        self.assertTrue(torch.allclose(dist.log_prob(low), expected_low))
        self.assertTrue(torch.allclose(dist.log_prob(high), expected_high))

    def test_rollout_and_ppo_evaluation_log_probs_match(self):
        torch.manual_seed(11)
        policy = AsymmetricActorCritic(
            AsymmetricModelConfig(init_log_std=-3.0)
        )
        actor_obs = torch.randn(32, 133)
        critic_obs = torch.randn(32, 153)

        with torch.no_grad():
            actions, rollout_log_prob, _ = policy.act(
                actor_obs, critic_obs, action_clip=1.0
            )
            evaluated_log_prob, _, _ = policy.evaluate_actions(
                actor_obs,
                critic_obs,
                actions,
                action_clip=1.0,
            )

        self.assertTrue(torch.all(actions.abs() <= 1.0))
        self.assertTrue(
            torch.allclose(rollout_log_prob, evaluated_log_prob, atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
