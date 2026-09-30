from isaacgymenvs.tasks.SRLEvo.srl_real_bot import SRL_Real_Bot
from isaacgymenvs.tasks.SRLEvo.stable_gait_rewards import StableGaitRewardMixin


class SRL_Real_Bot_Concurrent_v2(StableGaitRewardMixin, SRL_Real_Bot):
    """SRL task exposing five complete 30-D frames to Concurrent_v2."""

    def __init__(self, cfg, *args, **kwargs):
        # Actor and critic both receive the simulator's 153-D full-state stack.
        # The Concurrent_v2 agent replaces the actor's four privileged values
        # in every newly appended frame before PPO sees the observation.
        cfg["env"]["srl_policy_obs_remove_ids"] = []
        cfg["env"]["append_current_privileged_obs"] = False
        super().__init__(cfg, *args, **kwargs)

        if self.num_obs != 153 or self.srl_full_obs_size != 153:
            raise RuntimeError(
                "Concurrent_v2 requires 153-D raw actor observations and "
                "153-D critic states; got {} and {}".format(
                    self.num_obs, self.srl_full_obs_size
                )
            )
