# 该代码是用于加了柔性关节的xml的actor和estimator训练的

from isaacgymenvs.tasks.SRLEvo.srl_real_bot_compliant import (
    SRL_Real_Bot_Compliant,
)


class SRL_Real_Bot_Compliant_Concurrent(SRL_Real_Bot_Compliant):
    """Concurrent-estimator variant of the compliant SRL model."""

    def __init__(self, cfg, *args, **kwargs):
        cfg["env"]["srl_policy_obs_remove_ids"] = [0, 1, 2, 3]
        cfg["env"]["append_current_privileged_obs"] = True
        super().__init__(cfg, *args, **kwargs)

        if self.num_obs != 137 or self.srl_full_obs_size != 153:
            raise RuntimeError(
                "Compliant concurrent task requires 137-D actor observations "
                "and 153-D critic states; got {} and {}".format(
                    self.num_obs, self.srl_full_obs_size
                )
            )
