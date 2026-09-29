import torch
import torch.nn.functional as F

from isaacgymenvs.learning.SRLEvo.concurrent_privileged_estimator import (
    ConcurrentPrivilegedEstimator,
)


class ConcurrentV2Estimator(ConcurrentPrivilegedEstimator):
    """Explicit-state estimator with an optional temporal consistency loss."""

    def normalized_losses(
        self,
        history,
        target,
        previous_estimate=None,
        previous_target=None,
        temporal_valid=None,
    ):
        self._validate(history, target)
        target_std = self.target_std()
        normalized_target = (target - self.target_mean) / target_std
        normalized_prediction = self.forward_normalized(history)
        explicit_loss = F.mse_loss(normalized_prediction, normalized_target)

        temporal_loss = explicit_loss.new_zeros(())
        if (
            previous_estimate is not None
            and previous_target is not None
            and temporal_valid is not None
        ):
            valid = temporal_valid.reshape(-1).bool()
            if valid.any():
                previous_prediction = (
                    previous_estimate[valid] - self.target_mean
                ) / target_std
                predicted_delta = normalized_prediction[valid] - previous_prediction
                target_delta = (
                    target[valid] - previous_target[valid]
                ) / target_std
                temporal_loss = F.mse_loss(predicted_delta, target_delta)

        return {
            "explicit": explicit_loss,
            "temporal": temporal_loss,
        }
