"""Minibatch-time latent reconstruction for the feed-forward rl_games actor."""

from contextlib import contextmanager

import torch
from rl_games.algos_torch import torch_ext


class LatentPPORollout:
    """Env-major storage flattens without another horizon-sized history copy."""

    def __init__(self, horizon, num_envs, history_len, input_dim, device):
        prefix = (num_envs, horizon)
        specs = {
            "history": ((history_len, input_dim), torch.float32),
            "mirrored_history": ((history_len, input_dim), torch.float32),
            "truth": ((4,), torch.float32),
            "use_estimator": ((), torch.bool),
            "com_cop": ((3,), torch.float32),
            "contacts": ((2,), torch.float32),
            "next_frame": ((input_dim,), torch.float32),
            "mirrored_next_frame": ((input_dim,), torch.float32),
            "next_valid": ((), torch.bool),
        }
        self.horizon = horizon
        self.data = {key: torch.zeros(prefix + shape, dtype=dtype, device=device)
                     for key, (shape, dtype) in specs.items()}
        self.step = 0

    @torch.no_grad()
    def record(self, history, mirrored_history, truth, use_estimator, com_cop, contacts):
        if self.step >= self.horizon:
            raise RuntimeError("Latent rollout exceeds PPO horizon")
        for key, value in (("history", history), ("mirrored_history", mirrored_history),
                           ("truth", truth), ("use_estimator", use_estimator),
                           ("com_cop", com_cop), ("contacts", contacts)):
            target = self.data[key][:, self.step]
            if target.shape != value.shape:
                raise RuntimeError("Latent rollout shape mismatch for {}: {} != {}".format(
                    key, target.shape, value.shape))
            target.copy_(value)
        self.data["next_valid"][:, self.step] = False
        self.step += 1

    @torch.no_grad()
    def record_next(self, frame, mirrored_frame, dones):
        index = self.step - 1
        if index < 0:
            raise RuntimeError("Missing current latent transition")
        self.data["next_frame"][:, index].copy_(frame)
        self.data["mirrored_next_frame"][:, index].copy_(mirrored_frame)
        # Exclude both falls and timeouts; never learn a reset-to-next-episode jump.
        self.data["next_valid"][:, index].copy_(~dones.reshape(-1).bool())

    def flattened(self):
        if self.step != self.horizon:
            raise RuntimeError("Incomplete latent PPO rollout")
        return {"latent_" + key: value.flatten(0, 1) for key, value in self.data.items()}


@contextmanager
def differentiable_actor_normalization(model):
    """Bypass BaseModelNetwork.norm_obs's no_grad, not its normalization math.

    Only the transient instance method is changed, never the module hierarchy or
    state_dict. All statistics remain fixed until the PPO epoch has completed.
    """
    had_override = "norm_obs" in model.__dict__
    original = model.__dict__.get("norm_obs")

    def normalize(observation):
        if not model.normalize_input:
            return observation
        rms = model.running_mean_std
        if observation.ndim != 2 or not hasattr(rms, "running_var"):
            raise RuntimeError("Joint latent PPO requires flat vector RunningMeanStd")
        scale = torch.sqrt(rms.running_var.detach().float() + rms.epsilon)
        if rms.norm_only:
            return observation / scale
        return ((observation - rms.running_mean.detach().float()) / scale).clamp(-5.0, 5.0)

    model.norm_obs = normalize
    try:
        yield
    finally:
        if had_override:
            model.norm_obs = original
        else:
            del model.norm_obs


def gradient_norm(loss, parameters, scale=1.0):
    parameters = list(parameters)
    if not loss.requires_grad:
        return loss.new_zeros(())
    gradients = torch.autograd.grad(loss * scale, parameters, retain_graph=True, allow_unused=True)
    return torch.stack([(value.detach().float() / scale).square().sum() for value in gradients
                        if value is not None] + [loss.new_zeros((), dtype=torch.float32)]).sum().sqrt()


class LatentPPOUpdateMixin:
    """Actor Adam and one encoder Adam step from the same combined backward."""

    def _joint_actor_observation(self, batch, mirrored=False):
        prefix = "mirrored_" if mirrored else ""
        history = batch["latent_" + prefix + "history"]
        output = self.concurrent_estimator(history)
        truth = batch["latent_truth"]
        if mirrored:
            truth = truth * self._mirror_sign
        selected = torch.where(batch["latent_use_estimator"].reshape(-1, 1),
                               output["explicit"].detach(), truth)
        deployable = batch["obs_mirrored" if mirrored else "obs"][:, :self.deployable_obs_dim]
        return torch.cat((deployable, selected, output["latent"]), dim=-1), output

    def _joint_auxiliary_loss(self, batch):
        history = torch.cat((batch["latent_history"], batch["latent_mirrored_history"]))
        truth = batch["latent_truth"]
        explicit = torch.cat((truth, truth * self._mirror_sign))
        com = batch["latent_com_cop"]
        contacts = batch["latent_contacts"]
        losses = self.concurrent_estimator.compute_losses(
            history, explicit,
            com_cop_target=(torch.cat((com, com * com.new_tensor([1., -1., 1.])))
                            if self.use_com_cop_auxiliary else None),
            contact_target=(torch.cat((contacts, contacts[:, [1, 0]]))
                            if self.use_contact_auxiliary else None),
        )
        valid = batch["latent_next_valid"]
        if self.use_next_observation_loss and valid.any():
            next_losses = self.concurrent_estimator.compute_losses(
                torch.cat((batch["latent_history"][valid], batch["latent_mirrored_history"][valid])),
                torch.cat((truth[valid], truth[valid] * self._mirror_sign)),
                next_target=torch.cat((batch["latent_next_frame"][valid],
                                       batch["latent_mirrored_next_frame"][valid])),
            )
            losses["next_observation"] = next_losses["next_observation"]
        else:
            losses["next_observation"] = history.new_zeros(())
        total = history.new_zeros(())
        for name, value in losses.items():
            coefficient = self.loss_weights.get(name, 0.0)
            if coefficient != 0.0:
                total = total + coefficient * value
        return total, losses

    def _joint_forward(self, observation, actions=None, is_train=True):
        with differentiable_actor_normalization(self.model):
            return self.model({"obs": self._preproc_obs(observation),
                               "prev_actions": actions, "is_train": is_train,
                               "rnn_states": None})

    def _calc_joint_latent_gradients(self, batch):
        # Central critic has its own truth-only dataset and optimizer. Its loss
        # must not reach the encoder through the unused actor-side value head.
        if self.is_rnn or self.multi_gpu or not self.has_central_value:
            raise RuntimeError("Joint latent PPO requires feed-forward, single-GPU asymmetric PPO")
        self.optimizer.zero_grad(set_to_none=True)
        self.concurrent_estimator_optimizer.zero_grad(set_to_none=True)
        self.concurrent_estimator.train()
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            actor_obs, output = self._joint_actor_observation(batch)
            with torch.no_grad():
                mirrored_obs, _ = self._joint_actor_observation(batch, mirrored=True)
                mirrored_mu = self._joint_forward(mirrored_obs, is_train=False)["mus"]
            result = self._joint_forward(actor_obs, batch["actions"])
            neglogp, mu, sigma = result["prev_neglogp"], result["mus"], result["sigmas"]
            a_loss = self.actor_loss_func(batch["old_logp_actions"], neglogp,
                                          batch["advantages"], self.ppo, self.e_clip).mean()
            entropy = result["entropy"].mean()
            if self.bound_loss_type == "regularisation":
                b_loss = self.reg_loss(mu).mean()
            elif self.bound_loss_type == "bound":
                b_loss = self.bound_loss(mu).mean()
            else:
                b_loss = mu.new_zeros(())
            sym_loss = self.sym_loss(mu, mirrored_mu)["sym_loss"].mean()
            policy_loss = (a_loss - self.entropy_coef * entropy
                           + (self.bounds_loss_coef or 0.0) * b_loss
                           + (self.a_sym_loss_coef or 0.0) * sym_loss)
            aux_loss, losses = self._joint_auxiliary_loss(batch)
            loss = (policy_loss + self.latent_ppo_aux_loss_coef * aux_loss
                    if self.latent_ppo_aux_loss_coef != 0.0 else policy_loss)

        probe = (not self._joint_gradient_probed and
                 (self._joint_need_initial_probe or
                  int(self.epoch_num) % self.latent_ppo_grad_diagnostics_every == 0))
        if probe:
            self._joint_gradient_probed = True
            self._joint_need_initial_probe = False
            scale = self.scaler.get_scale()
            self._joint_metric("ppo_encoder_grad_norm", gradient_norm(a_loss, self.concurrent_estimator.encoder.parameters(), scale))
            self._joint_metric("ppo_latent_head_grad_norm", gradient_norm(a_loss, self.concurrent_estimator.latent_head.parameters(), scale))
            self._joint_metric("aux_encoder_grad_norm", gradient_norm(
                self.latent_ppo_aux_loss_coef * aux_loss, self.concurrent_estimator.encoder.parameters(), scale))
            print("[latent gradients] ppo_encoder={:.3e} ppo_latent_head={:.3e} aux_encoder={:.3e}".format(
                self._joint_metrics["ppo_encoder_grad_norm"][-1],
                self._joint_metrics["ppo_latent_head_grad_norm"][-1],
                self._joint_metrics["aux_encoder_grad_norm"][-1]))
        self.scaler.scale(loss).backward()
        # Both optimizers consume one common backward graph; neither can update
        # shared parameters before the other loss has contributed its gradients.
        self.scaler.unscale_(self.optimizer)
        self.scaler.unscale_(self.concurrent_estimator_optimizer)
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        torch.nn.utils.clip_grad_norm_(self.concurrent_estimator.parameters(), self.estimator_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.step(self.concurrent_estimator_optimizer)
        self.scaler.update()

        with torch.no_grad():
            kl = torch_ext.policy_kl(mu.detach(), sigma.detach(), batch["latent_old_mu"],
                                     batch["latent_old_sigma"], True)
            explicit_mse = (output["explicit"] - batch["latent_truth"]).square().mean()
            self._joint_metric("latent_std_mean", output["latent"].float().std(0, unbiased=False).mean())
            self._joint_metric("explicit_mse", explicit_mse)
            self._joint_metric("next_observation_mse", losses["next_observation"])
            self._joint_metric("aux_total", aux_loss)
            for name, value in losses.items():
                self._joint_metric(name, value)
        self.diagnostics.mini_batch(self, {"values": batch["old_values"],
            "returns": batch["returns"], "new_neglogp": neglogp,
            "old_neglogp": batch["old_logp_actions"], "masks": None}, self.e_clip, 0)
        self.train_result = (a_loss.detach(), mu.new_zeros(()), entropy.detach(), kl,
                             self.last_lr, 1.0, mu.detach(), sigma.detach(),
                             b_loss.detach(), sym_loss.detach())

    def _joint_metric(self, name, value):
        self._joint_metrics.setdefault(name, []).append(float(value.detach()))
