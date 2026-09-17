import torch


class EstimatedObservationAdapter(object):
    """Build causal actor observations from a frozen privileged estimator."""

    full_frame_dim = 30
    frame_stack = 5
    command_dim = 3
    privileged_dim = 4

    def __init__(self, model, num_envs, device):
        self.model = model
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.observation_dim = (
            self.full_frame_dim * self.frame_stack + self.command_dim
        )
        if model.input_dim != self.full_frame_dim - self.privileged_dim:
            raise ValueError(
                "Estimator input_dim={} is incompatible with a {}D frame".format(
                    model.input_dim, self.full_frame_dim
                )
            )
        if model.output_dim != self.privileged_dim:
            raise ValueError(
                "Estimator output_dim={} must equal {}".format(
                    model.output_dim, self.privileged_dim
                )
            )

        self.input_history = torch.zeros(
            self.num_envs,
            model.history_len,
            model.input_dim,
            device=self.device,
        )
        # Environment frame order is newest first.
        self.estimated_privileged_history = torch.zeros(
            self.num_envs,
            self.frame_stack,
            model.output_dim,
            device=self.device,
        )
        self.mirror_privileged_sign = torch.tensor(
            [1.0, 1.0, -1.0, 1.0],
            device=self.device,
            dtype=self.estimated_privileged_history.dtype,
        )
        self.initialized = False

    def _validate_observation(self, observation, name):
        if observation.ndim != 2 or tuple(observation.shape) != (
            self.num_envs,
            self.observation_dim,
        ):
            raise RuntimeError(
                "{} must have shape [{}, {}], got {}".format(
                    name,
                    self.num_envs,
                    self.observation_dim,
                    tuple(observation.shape),
                )
            )

    @staticmethod
    def _validate_alpha(alpha):
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("estimator alpha must be in [0, 1]")
        return alpha

    def _update_estimate(self, observation, first):
        first = first.to(device=self.device, dtype=torch.bool).reshape(-1).clone()
        if first.shape[0] != self.num_envs:
            raise RuntimeError(
                "first must contain {} entries, got {}".format(
                    self.num_envs, first.shape[0]
                )
            )

        frames = observation[:, : self.full_frame_dim * self.frame_stack].reshape(
            self.num_envs, self.frame_stack, self.full_frame_dim
        )
        current_input = frames[:, 0, self.privileged_dim :]

        if not self.initialized:
            self.input_history[:] = current_input.unsqueeze(1)
            first[:] = True
            self.initialized = True
        else:
            self.input_history[:, :-1] = self.input_history[:, 1:].clone()
            self.input_history[:, -1] = current_input
            if first.any():
                self.input_history[first] = current_input[first].unsqueeze(1).expand(
                    -1, self.model.history_len, -1
                )

        estimate = self.model(self.input_history)
        self.estimated_privileged_history[:, 1:] = (
            self.estimated_privileged_history[:, :-1].clone()
        )
        self.estimated_privileged_history[:, 0] = estimate
        if first.any():
            self.estimated_privileged_history[first] = estimate[first].unsqueeze(
                1
            ).expand(-1, self.frame_stack, -1)
        return frames, estimate

    def _replace_privileged(self, observation, estimates, alpha):
        frames = observation[:, : self.full_frame_dim * self.frame_stack].reshape(
            self.num_envs, self.frame_stack, self.full_frame_dim
        )
        actor_frames = frames.clone()
        actor_frames[:, :, : self.privileged_dim] = torch.lerp(
            frames[:, :, : self.privileged_dim], estimates, alpha
        )
        return torch.cat(
            (
                actor_frames.reshape(
                    self.num_envs, self.full_frame_dim * self.frame_stack
                ),
                observation[:, -self.command_dim :],
            ),
            dim=-1,
        )

    def transform(self, observation, first, alpha=1.0):
        """Replace five privileged frames and return observation/current estimate."""
        self._validate_observation(observation, "observation")
        alpha = self._validate_alpha(alpha)
        _, estimate = self._update_estimate(observation, first)
        actor_observation = self._replace_privileged(
            observation, self.estimated_privileged_history, alpha
        )
        return actor_observation, estimate

    def transform_pair(self, observation, mirrored_observation, first, alpha=1.0):
        """Transform normal and mirrored observations using one causal estimate."""
        self._validate_observation(observation, "observation")
        self._validate_observation(mirrored_observation, "mirrored_observation")
        alpha = self._validate_alpha(alpha)
        _, estimate = self._update_estimate(observation, first)

        actor_observation = self._replace_privileged(
            observation, self.estimated_privileged_history, alpha
        )
        mirrored_estimates = (
            self.estimated_privileged_history * self.mirror_privileged_sign
        )
        mirrored_actor_observation = self._replace_privileged(
            mirrored_observation, mirrored_estimates, alpha
        )
        return actor_observation, mirrored_actor_observation, estimate
