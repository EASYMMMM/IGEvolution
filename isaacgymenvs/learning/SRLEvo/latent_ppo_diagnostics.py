"""Opt-in numerical probes; never modify policy inputs, losses or schedules."""

import json
import math
from pathlib import Path

import torch
from rl_games.algos_torch import torch_ext


def gaussian_kl_fp32(mu, sigma, old_mu, old_sigma):
    """KL(current || rollout), without rl_games' additive variance epsilon."""
    mu, sigma, old_mu, old_sigma = [x.detach().float() for x in (mu, sigma, old_mu, old_sigma)]
    log_ratio = sigma.log() - old_sigma.log()
    return (0.5 * (torch.expm1(2 * log_ratio) - 2 * log_ratio
                   + ((mu - old_mu) / old_sigma).square())).sum(-1)


def measured_optimizer_step(scaler, optimizer):
    """Observe whether GradScaler calls step, including zero-LR/no-grad steps."""
    had_override = "step" in optimizer.__dict__
    previous = optimizer.__dict__.get("step")
    original = optimizer.step
    executed = False

    def step(*args, **kwargs):
        nonlocal executed
        executed = True
        return original(*args, **kwargs)

    optimizer.step = step
    try:
        scaler.step(optimizer)
    finally:
        if had_override:
            optimizer.step = previous
        else:
            del optimizer.step
    return executed


def _gradient_vector(loss, parameters, scale):
    if not loss.requires_grad:
        return torch.cat([torch.zeros_like(p).float().reshape(-1) for p in parameters])
    gradients = torch.autograd.grad(loss * scale, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(g.detach().float() / scale if g is not None else torch.zeros_like(p).float())
                      .reshape(-1) for p, g in zip(parameters, gradients)])


def _cosine(left, right):
    denominator = left.norm() * right.norm()
    if not torch.isfinite(denominator):
        return left.new_tensor(float("nan"))
    return (left.dot(right) / denominator) if denominator > 0 else left.new_zeros(())


def _snapshot(optimizer):
    return [(p, p.detach().clone()) for group in optimizer.param_groups for p in group["params"]]


def _gradient_stats(snapshot):
    gradients = [p.grad.detach().float().reshape(-1) for p, _ in snapshot if p.grad is not None]
    device = snapshot[0][0].device
    flat = torch.cat(gradients) if gradients else torch.zeros(1, device=device)
    return {"grad_finite": torch.isfinite(flat).all().float(), "grad_norm_pre_clip": flat.norm(),
            "grad_present_fraction": float(len(gradients)) / len(snapshot)}


def _parameter_delta(snapshot):
    deltas = [(p.detach().float() - before.float()).reshape(-1) for p, before in snapshot]
    delta = torch.cat(deltas)
    reference_norm = torch.stack([before.float().square().sum() for _, before in snapshot]).sum().sqrt()
    return {"delta_norm": delta.norm(), "delta_relative": delta.norm() / reference_norm.clamp_min(1e-12),
            "delta_max_abs": delta.abs().max(), "parameters_finite": torch.stack([
                torch.isfinite(p).all() for p, _ in snapshot]).all().float()}


def _ratio_stats(old_neglogp, new_neglogp, clip):
    log_ratio = old_neglogp.detach().float() - new_neglogp.detach().float()
    ratio = log_ratio.exp().reshape(-1)
    finite = torch.isfinite(ratio)
    values = ratio[finite]
    if values.numel():
        quantiles = torch.quantile(values, values.new_tensor([.01, .5, .99]))
        mean = values.mean()
    else:
        quantiles = ratio.new_full((3,), float("nan"))
        mean = ratio.new_tensor(float("nan"))
    return {"ratio_mean_finite": mean, "ratio_p01_finite": quantiles[0],
            "ratio_p50_finite": quantiles[1], "ratio_p99_finite": quantiles[2],
            "ratio_nonfinite_fraction": (~finite).float().mean(),
            "ratio_abs_error_max": (ratio - 1).abs().max(),
            "clip_fraction": ((ratio < 1 - clip) | (ratio > 1 + clip)).float().mean()}


class LatentPPONumerics:
    def __init__(self, agent, every=10, samples=1024, tolerance=1e-4, output=None):
        if every < 1 or samples < 1 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Invalid latent PPO numerical diagnostic settings")
        self.agent, self.every, self.samples, self.tolerance = agent, every, samples, tolerance
        self.output = Path(output) if output else None
        self.last_epoch = None
        self.pending_update = False
        self.record_id = 0
        if self.output:
            self.output.parent.mkdir(parents=True, exist_ok=True)
        print("Latent numerical diagnostics enabled: every={} samples={} output={}".format(
            every, samples, self.output or "TensorBoard/W&B only"))

    def _emit(self, phase, metrics):
        numbers = {key: float(value.detach()) if torch.is_tensor(value) else float(value)
                   for key, value in metrics.items()}
        for key, value in numbers.items():
            self.agent._joint_metric("numerics/" + phase + "/" + key, value)
        if self.output:
            self.record_id += 1
            record = {"phase": phase, "epoch": int(self.agent.epoch_num),
                      "frame": int(self.agent.frame), "record_id": self.record_id,
                      "mixed_precision": bool(self.agent.mixed_precision),
                      "metrics": {k: v if math.isfinite(v) else None for k, v in numbers.items()},
                      "nonfinite_metrics": [k for k, v in numbers.items() if not math.isfinite(v)]}
            with self.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        return numbers

    @torch.no_grad()
    def prepare(self, values):
        """Called before central-value training or any actor/encoder update."""
        epoch = int(self.agent.epoch_num)
        if self.last_epoch is not None and (self.last_epoch == epoch or epoch % self.every != 0):
            return
        self.last_epoch = epoch
        self.pending_update = True
        size = values["obs"].shape[0]
        ids = torch.linspace(0, size - 1, min(self.samples, size), device=values["obs"].device).long()
        batch = {key: value[ids] for key, value in values.items()
                 if torch.is_tensor(value) and value.ndim and value.shape[0] == size}
        # Rollout uses FP32. Also probe training precision separately, without
        # sampling actions or consuming RNG. Neither path updates RMS buffers.
        with torch.cuda.amp.autocast(enabled=False):
            obs, output = self.agent._joint_actor_observation(batch)
            mirrored_obs, _ = self.agent._joint_actor_observation(batch, mirrored=True)
            result = self.agent._joint_forward(obs, batch["actions"])
        start = self.agent.deployable_obs_dim
        explicit_end = start + self.agent.estimator_output_dim
        metrics = {
            "sample_count": len(ids),
            "explicit_prediction_max_abs": (output["explicit"] - batch["latent_explicit_prediction"]).abs().max(),
            "actor_explicit_max_abs": (obs[:, start:explicit_end] - batch["obs"][:, start:explicit_end]).abs().max(),
            "latent_max_abs": (obs[:, explicit_end:] - batch["obs"][:, explicit_end:]).abs().max(),
            "mirrored_actor_max_abs": (mirrored_obs - batch["obs_mirrored"]).abs().max(),
            "mu_max_abs": (result["mus"] - batch["latent_old_mu"]).abs().max(),
            "sigma_max_abs": (result["sigmas"] - batch["latent_old_sigma"]).abs().max(),
            "neglogp_max_abs": (result["prev_neglogp"] - batch["old_logp_actions"]).abs().max(),
        }
        errors = list(metrics.values())[1:]
        metrics["consistency_pass"] = float(all(torch.isfinite(x).item() and x.item() <= self.tolerance
                                                for x in errors))
        metrics.update(_ratio_stats(batch["old_logp_actions"], result["prev_neglogp"], self.agent.e_clip))
        metrics["kl_fp32"] = gaussian_kl_fp32(result["mus"], result["sigmas"],
                                               batch["latent_old_mu"], batch["latent_old_sigma"]).mean()
        with torch.cuda.amp.autocast(enabled=self.agent.mixed_precision):
            train_obs, _ = self.agent._joint_actor_observation(batch)
            train_result = self.agent._joint_forward(train_obs, batch["actions"])
        metrics["train_precision_neglogp_max_abs"] = (train_result["prev_neglogp"].float() - batch["old_logp_actions"]).abs().max()
        metrics["train_precision_mu_max_abs"] = (train_result["mus"].float() - batch["latent_old_mu"]).abs().max()
        record = self._emit("pre_update", metrics)
        print("[latent numerics pre] epoch={} pass={} explicit={:.3e} latent={:.3e} mu={:.3e} logp={:.3e} ratio_err={:.3e}".format(
            epoch, int(record["consistency_pass"]), record["explicit_prediction_max_abs"], record["latent_max_abs"],
            record["mu_max_abs"], record["neglogp_max_abs"], record["ratio_abs_error_max"]))

    @torch.no_grad()
    def _input_metrics(self, observation):
        latent_start = self.agent.deployable_obs_dim + self.agent.estimator_output_dim
        raw = observation.detach().float()[:, latent_start:]
        model = self.agent.model
        if model.normalize_input:
            rms = model.running_mean_std
            mean = rms.running_mean.detach().float()[latent_start:]
            var = rms.running_var.detach().float()[latent_start:]
            normalized = (raw if rms.norm_only else raw - mean) / torch.sqrt(var + rms.epsilon)
            clipped = normalized if rms.norm_only else normalized.clamp(-5, 5)
            clipped_fraction = (normalized.abs() > 5).float().mean() if not rms.norm_only else raw.new_zeros(())
            stats = {"rms_mean_abs": mean.abs().mean(), "rms_var_mean": var.mean(), "rms_count": rms.count}
        else:
            normalized = clipped = raw
            clipped_fraction = raw.new_zeros(())
            stats = {"rms_mean_abs": 0., "rms_var_mean": 1., "rms_count": 0.}
        stats.update({"latent_raw_mean": raw.mean(), "latent_raw_std": raw.std(0, unbiased=False).mean(),
                      "latent_raw_max_abs": raw.abs().max(),
                      "latent_normalized_mean": normalized.mean(),
                      "latent_normalized_std": normalized.std(0, unbiased=False).mean(),
                      "latent_normalized_max_abs": normalized.abs().max(),
                      "latent_clipped_std": clipped.std(0, unbiased=False).mean(),
                      "latent_clip_fraction": clipped_fraction})
        return stats

    def begin_update(self, batch, observation, result, actor_loss, symmetry_loss, policy_loss, auxiliary_loss):
        """Probe the actual first minibatch graph, not a constructed actor loss."""
        if not self.pending_update:
            return None
        self.pending_update = False
        parameters = list(self.agent.concurrent_estimator.encoder.parameters())
        scale = self.agent.scaler.get_scale()
        gradients = {name: _gradient_vector(loss, parameters, scale) for name, loss in (
            ("surrogate", actor_loss), ("symmetry", symmetry_loss),
            ("policy", policy_loss), ("auxiliary", auxiliary_loss))}
        metrics = self._input_metrics(observation)
        metrics.update({name + "_encoder_grad_norm": grad.norm() for name, grad in gradients.items()})
        metrics.update({"surrogate_aux_cosine": _cosine(gradients["surrogate"], gradients["auxiliary"]),
                        "policy_aux_cosine": _cosine(gradients["policy"], gradients["auxiliary"]),
                        "surrogate_symmetry_cosine": _cosine(gradients["surrogate"], gradients["symmetry"]),
                        "surrogate_loss": actor_loss.detach(), "weighted_symmetry_loss": symmetry_loss.detach(),
                        "policy_loss": policy_loss.detach(), "weighted_auxiliary_loss": auxiliary_loss.detach(),
                        "scale_before": scale, "actor_lr": self.agent.optimizer.param_groups[0]["lr"],
                        "encoder_lr": self.agent.concurrent_estimator_optimizer.param_groups[0]["lr"],
                        "mu_mean_abs": result["mus"].detach().float().abs().mean(),
                        "mu_max_abs": result["mus"].detach().float().abs().max(),
                        "sigma_mean": result["sigmas"].detach().float().mean(),
                        "sigma_min": result["sigmas"].detach().float().min(),
                        "sigma_max": result["sigmas"].detach().float().max()})
        metrics.update(_ratio_stats(batch["old_logp_actions"], result["prev_neglogp"], self.agent.e_clip))
        mu, sigma = result["mus"].detach().float(), result["sigmas"].detach().float()
        old_mu, old_sigma = batch["latent_old_mu"].float(), batch["latent_old_sigma"].float()
        library_kl = torch_ext.policy_kl(mu, sigma, old_mu, old_sigma, False)
        metrics.update({"kl_fp32": gaussian_kl_fp32(mu, sigma, old_mu, old_sigma).mean(),
                        "kl_library_fp32": library_kl.mean(),
                        "kl_library_training_dtype": torch_ext.policy_kl(
                            result["mus"].detach(), result["sigmas"].detach(),
                            batch["latent_old_mu"], batch["latent_old_sigma"], True),
                        "kl_library_negative_fraction": (library_kl < 0).float().mean()})
        state = {"metrics": metrics, "actor": _snapshot(self.agent.optimizer),
                 "encoder": _snapshot(self.agent.concurrent_estimator_optimizer), "batch": batch,
                 "mu_before": mu.clone(), "neglogp_before": result["prev_neglogp"].detach().float().clone()}
        # Exclude log_std and the unused actor-side value head. The final prefix
        # also covers the small actor used by the real-PPO regression harness.
        mean_ids = {id(parameter) for name, parameter in self.agent.model.named_parameters()
                    if name.startswith(("a2c_network.actor_mlp.", "a2c_network.mu.", "a2c_network.actor."))}
        mean_snapshot = [(parameter, before) for parameter, before in state["actor"] if id(parameter) in mean_ids]
        if mean_snapshot:
            state["actor_mean"] = mean_snapshot
        return state

    def after_unscale(self, state):
        for name in ("actor", "encoder", "actor_mean"):
            if name not in state:
                continue
            state["metrics"].update({name + "_" + key: value for key, value in _gradient_stats(state[name]).items()})

    @torch.no_grad()
    def finish_update(self, state, actor_stepped, encoder_stepped):
        metrics = state["metrics"]
        for name, stepped in (("actor", actor_stepped), ("encoder", encoder_stepped)):
            metrics.update({name + "_" + key: value for key, value in _parameter_delta(state[name]).items()})
            metrics[name + "_step_executed"] = float(stepped)
            metrics[name + "_amp_skipped"] = float(self.agent.scaler.is_enabled() and not stepped)
        if "actor_mean" in state:
            metrics.update({"actor_mean_" + key: value for key, value in _parameter_delta(state["actor_mean"]).items()})
        metrics["scale_after"] = self.agent.scaler.get_scale()
        # The first pre-step ratio should be one. Replay after the real step as
        # well, to measure how much this update actually changes the policy.
        batch = state["batch"]
        with torch.cuda.amp.autocast(enabled=self.agent.mixed_precision):
            observation, _ = self.agent._joint_actor_observation(batch)
            result = self.agent._joint_forward(observation, batch["actions"])
        metrics.update({"post_" + key: value for key, value in _ratio_stats(
            batch["old_logp_actions"], result["prev_neglogp"], self.agent.e_clip).items()})
        metrics["post_kl_fp32"] = gaussian_kl_fp32(
            result["mus"], result["sigmas"], batch["latent_old_mu"], batch["latent_old_sigma"]).mean()
        metrics["post_kl_library_fp32"] = torch_ext.policy_kl(
            result["mus"].float(), result["sigmas"].float(),
            batch["latent_old_mu"].float(), batch["latent_old_sigma"].float(), True)
        metrics["post_kl_library_training_dtype"] = torch_ext.policy_kl(
            result["mus"], result["sigmas"], batch["latent_old_mu"], batch["latent_old_sigma"], True)
        metrics["post_mu_delta_max_abs"] = (result["mus"].float() - state["mu_before"]).abs().max()
        metrics["post_neglogp_delta_max_abs"] = (result["prev_neglogp"].float() - state["neglogp_before"]).abs().max()
        record = self._emit("update", metrics)
        print("[latent numerics update] epoch={} clip={:.3f} grad(policy/aux)={:.3e}/{:.3e} delta(actor/encoder)={:.3e}/{:.3e} mean_delta={:.3e} stepped={}/{} post_kl={:.3e}".format(
            int(self.agent.epoch_num), record["latent_clip_fraction"], record["policy_encoder_grad_norm"],
            record["auxiliary_encoder_grad_norm"], record["actor_delta_norm"], record["encoder_delta_norm"],
            record.get("actor_mean_delta_norm", float("nan")),
            int(record["actor_step_executed"]), int(record["encoder_step_executed"]), record["post_kl_fp32"]))
