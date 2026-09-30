from isaacgymenvs.tasks.SRLEvo.srl_real_bot import SRL_Real_Bot
from isaacgymenvs.tasks.SRLEvo.stable_gait_rewards import StableGaitRewardMixin


class SRL_Real_Bot_Concurrent(StableGaitRewardMixin, SRL_Real_Bot):
    """SRL task variant for concurrent policy and state-estimator training."""

    def __init__(self, cfg, *args, **kwargs):
        cfg["env"]["srl_policy_obs_remove_ids"] = [0, 1, 2, 3]
        cfg["env"]["append_current_privileged_obs"] = True
        super().__init__(cfg, *args, **kwargs)

        if self.num_obs != 137 or self.srl_full_obs_size != 153:
            raise RuntimeError(
                "Concurrent SRL task requires 137-D actor observations and "
                "153-D critic states; got {} and {}".format(
                    self.num_obs, self.srl_full_obs_size
                )
            )
