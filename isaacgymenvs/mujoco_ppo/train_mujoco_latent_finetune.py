"""MuJoCo latent PPO fine-tuning; shares concurrent sampling and task curricula."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import mujoco
import torch
from torch.nn import functional as F

from mujoco_ppo import train_mujoco_concurrent_finetune as shared
from mujoco_ppo.models import safe_torch_load
from mujoco_ppo.models_concurrent_latent import load_latent_checkpoint, mirror_explicit, mirror_frames


@dataclass
class LatentPPOConfig(shared.ConcurrentPPOConfig):
    control_backend: str = "torque_pd"
    dof_vel_obs_scale: float = 0.05
    actor_input_mode: str = "estimated"
    estimator_only_updates: int = 0
    latent_ppo_update_encoder: bool = True
    latent_explicit_loss_coef: float = 1.0
    latent_next_observation_loss_coef: float = 1.0
    latent_l2_loss_coef: float = 1e-4
    latent_aux_loss_coef: float = 1.0
    latent_mirror_supervision: bool = True
    latent_gradient_diagnostics: bool = True
    latent_update_normalization: bool = False
    latent_encoder_max_grad_norm: float = 1.0
    save_dir: str = "mujoco_ppo/runs/latent"


def configure_startup_stage3_mix(cfg):
    """Apply a task-only preset before CLI overrides, keeping the saved controller."""
    if cfg.mixed_task_stage3_initial_pose_randomization_enable is None:
        cfg.mixed_task_stage3_initial_pose_randomization_enable = cfg.initial_pose_randomization_enable
    cfg.mixed_task_enable = True
    cfg.three_task_enable = False
    cfg.mixed_task_inplace_fraction = 0.5
    cfg.mixed_task_action_scale = cfg.action_scale
    cfg.inplace_startup_task_enable = True
    cfg.initial_pose_randomization_enable = True
    cfg.inplace_startup_scenario_probabilities = (0.0, 1.0, 0.0)
    cfg.inplace_startup_initial_height_range = (0.93, 1.0)
    cfg.inplace_startup_height_hold_time_range = (1.0, 2.0)
    cfg.inplace_startup_height_ramp_time_range = (2.0, 4.0)
    cfg.inplace_startup_first_target_height_range = (0.85, 0.95)
    cfg.inplace_startup_support_sync_with_height_ramp = True
    cfg.inplace_startup_randomize_height_after_unload = False
    cfg.inplace_startup_tether_release_delay = 1.0
    cfg.inplace_startup_disturbance_after_unload_only = True
    cfg.startup_support_enable = cfg.startup_support_in_evaluation = True
    cfg.startup_support_curriculum_enable = False
    cfg.startup_support_mode_probabilities = (0.8, 0.0, 0.0, 0.2)
    cfg.startup_support_fraction_range = (0.2, 0.5)
    cfg.startup_support_unload_time_range = (5.0, 10.0)
    cfg.startup_support_hold_until_unloaded = True
    cfg.startup_support_post_unload_hold_time = 1.0
    cfg.startup_inplace_time = cfg.startup_ramp_time = 0.0
    cfg.startup_tether_enable = True
    cfg.startup_tether_stiffness_1mg_range = (150.0, 200.0)
    cfg.startup_tether_damping_range = (10.0, 20.0)
    cfg.dr_curriculum_support_mode = "unload_to_zero"


class LatentRolloutBuffer(shared.ConcurrentRolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = (self.steps, self.num_envs)
        device = self.rewards.device
        self.use_estimate = torch.zeros(shape, dtype=torch.bool, device=device)
        self.next_frame = torch.zeros((*shape, 26), device=device)
        self.next_valid = torch.zeros(shape, dtype=torch.bool, device=device)

    def batches_with_weights(self, minibatch_size, sample_weights=None):
        size = self.steps * self.num_envs
        weights = None if sample_weights is None else sample_weights.reshape(size)
        arrays = (self.actor_obs.reshape(size, -1), self.mirrored_actor_obs.reshape(size, -1),
                  self.critic_obs.reshape(size, -1), self.actions.reshape(size, -1),
                  self.logprobs.reshape(size), self.advantages.reshape(size), self.returns.reshape(size),
                  self.values.reshape(size), self.estimator_history.reshape(size, *self.estimator_history.shape[2:]),
                  self.estimator_target.reshape(size, 4), self.use_estimate.reshape(size),
                  self.next_frame.reshape(size, 26), self.next_valid.reshape(size))
        indices = torch.randperm(size, device=self.rewards.device)
        for start in range(0, size, minibatch_size):
            ids = indices[start:start + minibatch_size]
            batch = tuple(array[ids] for array in arrays)
            yield (*batch[:8], None if weights is None else weights[ids], *batch[8:])


def _grad_norm(parameters, gradients=None):
    values = [p.grad for p in parameters] if gradients is None else gradients
    squares = [g.detach().float().square().sum() for g in values if g is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


class LatentTrainingBackend:
    buffer_class = LatentRolloutBuffer

    def validate(self, cfg):
        if cfg.control_backend == "position_target":
            model = mujoco.MjModel.from_xml_path(cfg.xml_path)
            if np.any(model.actuator_biasprm[:, 1] >= 0):
                raise ValueError(
                    "position_target requires position-servo actuators, but this XML contains torque/other motors. "
                    "Use control_backend=torque_pd; never send target angles directly to torque motors."
                )
        saved = safe_torch_load(cfg.checkpoint_path, map_location="cpu")
        if "concurrent_latent_model" not in saved:
            raise ValueError("A latent checkpoint is required; concurrent checkpoints are incompatible")
        model_config = saved.get("concurrent_latent_config", {})
        cfg.estimator_history_len = int(model_config.get("history_len", 10))
        if cfg.resume_optimizer:
            if saved.get("checkpoint_type") != "mujoco_concurrent_latent":
                raise ValueError("Cannot restore an Isaac optimizer into MuJoCo latent PPO; use resume_optimizer=false")
            previous = saved.get("config", {}).get("latent_ppo_update_encoder")
            if previous != cfg.latent_ppo_update_encoder:
                raise ValueError("Optimizer resume across latent training modes is not supported")
        if min(cfg.latent_explicit_loss_coef, cfg.latent_next_observation_loss_coef,
               cfg.latent_l2_loss_coef, cfg.latent_aux_loss_coef) < 0:
            raise ValueError("Latent loss coefficients must be nonnegative")
        if cfg.latent_encoder_max_grad_norm <= 0 or cfg.estimator_learning_rate <= 0:
            raise ValueError("Encoder gradient clip and learning rate must be positive")
        if cfg.update_epochs < 1 or (not cfg.latent_ppo_update_encoder and cfg.estimator_update_epochs < 1):
            raise ValueError("PPO and independent auxiliary update epochs must be positive")

    @staticmethod
    def make_policy(cfg, env):
        return load_latent_checkpoint(cfg.checkpoint_path, cfg.device, cfg.checkpoint_key)

    @staticmethod
    def layout(policy, cfg):
        return ("Latent layout: encoder {}x26 -> explicit 4 + latent 16; actor 153 -> 6; critic 153 -> 1; "
                "joint PPO encoder={}; encoder LR={:.2e}; encoder clip={:g}; FP32".format(
                    policy.encoder.history_len, cfg.latent_ppo_update_encoder,
                    cfg.estimator_learning_rate, cfg.latent_encoder_max_grad_norm))

    def setup(self, policy, cfg, optimizers):
        self.statistics = defaultdict(list)
        if cfg.resume_optimizer:
            saved = safe_torch_load(cfg.checkpoint_path, map_location=cfg.device)
            states = saved.get("optimizer_state_dict", {})
            if set(states) != set(optimizers):
                raise ValueError("Checkpoint optimizer groups do not match latent trainer")
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(states[name])
            print("Restored MuJoCo latent optimizer states (including saved learning rates)")
        # Encoder parameters belong only to the estimator optimizer, even in joint mode.
        owners = [id(p) for opt in optimizers.values() for group in opt.param_groups for p in group["params"]]
        if len(owners) != len(set(owners)):
            raise RuntimeError("A parameter is owned by more than one optimizer")

    @staticmethod
    def record_transition(buffer, modes, next_infos, terminated, truncated):
        index = buffer.ptr
        buffer.use_estimate[index] = torch.as_tensor(modes, device=buffer.rewards.device)
        buffer.next_frame[index] = torch.as_tensor(
            np.stack([info["deployable_obs"][:26] for info in next_infos]), device=buffer.rewards.device)
        buffer.next_valid[index] = torch.as_tensor(
            (np.asarray(terminated) == 0) & (np.asarray(truncated) == 0), device=buffer.rewards.device)

    def auxiliary_loss(self, encoder, history, truth, next_frame, valid, cfg):
        if cfg.latent_mirror_supervision:
            history = torch.cat((history, mirror_frames(history)))
            truth = torch.cat((truth, mirror_explicit(truth)))
            next_frame = torch.cat((next_frame, mirror_frames(next_frame)))
            valid = torch.cat((valid, valid))
        output = encoder(history)
        explicit_loss = F.mse_loss(output["explicit_normalized"],
                                  (truth - encoder.explicit_mean) / encoder._std(encoder.explicit_var))
        if valid.any():
            decoder_output = {key: value[valid] for key, value in output.items()}
            next_loss = F.mse_loss(encoder.decode(decoder_output),
                                  (next_frame[valid] - encoder.next_mean) / encoder._std(encoder.next_var))
        else:
            next_loss = output["latent"].sum() * 0
        l2 = output["latent"].square().mean()
        loss = cfg.latent_aux_loss_coef * (
            cfg.latent_explicit_loss_coef * explicit_loss
            + cfg.latent_next_observation_loss_coef * next_loss + cfg.latent_l2_loss_coef * l2)
        for name, value in (("explicit_loss", explicit_loss), ("next_observation_loss", next_loss),
                            ("latent_l2_loss", l2), ("aux_total_loss", loss),
                            ("explicit_mse", (output["explicit"] - truth).square().mean()),
                            ("latent_std_mean", output["latent"].std(0, unbiased=False).mean())):
            self.statistics[name].append(float(value.detach()))
        if valid.any():
            with torch.no_grad():
                prediction = encoder.decode(decoder_output) * encoder._std(encoder.next_var) + encoder.next_mean
                self.statistics["next_observation_mse"].append(float((prediction - next_frame[valid]).square().mean()))
        return loss

    def before_ppo(self, policy, buffer, cfg, optimizers):
        self.statistics.clear()
        history = buffer.estimator_history.reshape(-1, cfg.estimator_history_len, 26)
        truth = buffer.estimator_target.reshape(-1, 4)
        next_frame, valid = buffer.next_frame.reshape(-1, 26), buffer.next_valid.reshape(-1)
        self.normalization_data = (history, truth, next_frame, valid)
        # Keep rollout normalization fixed through PPO. Optional updates occur afterwards.
        if not cfg.latent_ppo_update_encoder:
            for _ in range(cfg.estimator_update_epochs):
                ids = torch.randperm(len(history), device=history.device)
                for start in range(0, len(ids), cfg.estimator_minibatch_size):
                    selected = ids[start:start + cfg.estimator_minibatch_size]
                    loss = self.auxiliary_loss(policy.encoder, history[selected], truth[selected],
                                               next_frame[selected], valid[selected], cfg)
                    optimizers["estimator"].zero_grad(set_to_none=True)
                    loss.backward()
                    self.statistics["aux_encoder_grad_norm"].append(_grad_norm(policy.encoder.encoder.parameters()))
                    torch.nn.utils.clip_grad_norm_(policy.encoder.parameters(), cfg.latent_encoder_max_grad_norm)
                    optimizers["estimator"].step()
        with torch.no_grad():
            estimate = policy.encoder(history)["explicit_normalized"]
            return float(F.mse_loss(estimate, (truth - policy.encoder.explicit_mean) / policy.encoder._std(policy.encoder.explicit_var)))

    def prepare_minibatch(self, policy, reference, batch, cfg):
        history, truth, modes, next_frame, valid = batch[9:]
        deployable, mirrored = batch[0][:, :133], batch[1][:, :133]
        with torch.no_grad():
            reference_obs, _ = reference.compose_actor_obs(deployable, history, truth, modes)
        context = {"history": history, "truth": truth, "next_frame": next_frame,
                   "valid": valid, "reference_obs": reference_obs}
        if cfg.latent_ppo_update_encoder:
            actor, mirror, _ = policy.compose_mirrored_actor_obs(deployable, mirrored, history, truth, modes)
            return actor, mirror, context
        return batch[0], batch[1], context

    def step_minibatch(self, policy, cfg, optimizers, actor_params, actor_loss, actor_enabled, context, ppo_loss=None):
        optimizers["actor"].zero_grad(set_to_none=True)
        optimizers["estimator"].zero_grad(set_to_none=True)
        auxiliary = None
        if cfg.latent_ppo_update_encoder:
            auxiliary = self.auxiliary_loss(policy.encoder, context["history"], context["truth"],
                                            context["next_frame"], context["valid"], cfg)
        if cfg.latent_gradient_diagnostics:
            policy_loss = actor_loss if ppo_loss is None else ppo_loss
            for name, parameters, loss in (
                ("ppo_encoder_grad_norm", list(policy.encoder.encoder.parameters()), policy_loss if actor_enabled and cfg.latent_ppo_update_encoder else None),
                ("ppo_latent_head_grad_norm", list(policy.encoder.latent_head.parameters()), policy_loss if actor_enabled and cfg.latent_ppo_update_encoder else None),
                ("aux_encoder_grad_norm", list(policy.encoder.encoder.parameters()), auxiliary),
            ):
                grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True) if loss is not None else []
                self.statistics[name].append(_grad_norm(parameters, grads))
        total = actor_loss if actor_enabled else None
        if auxiliary is not None:
            total = auxiliary if total is None else total + auxiliary
        if total is not None:
            total.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(policy.encoder.parameters(), cfg.latent_encoder_max_grad_norm)
            if actor_enabled:
                optimizers["actor"].step()
            if auxiliary is not None:
                optimizers["estimator"].step()

    def metrics(self):
        return {"concurrent_latent/" + name: float(np.mean(values))
                for name, values in self.statistics.items() if values}

    def after_ppo(self, policy, cfg):
        if cfg.latent_update_normalization:
            history, truth, next_frame, valid = self.normalization_data
            if cfg.latent_mirror_supervision:
                history = torch.cat((history, mirror_frames(history)))
                truth = torch.cat((truth, mirror_explicit(truth)))
                next_frame = torch.cat((next_frame, mirror_frames(next_frame)))
                valid = torch.cat((valid, valid))
            policy.encoder.update_normalization(history, truth, next_frame[valid])
        self.normalization_data = None


def train(cfg):
    return shared.train(cfg, backend=LatentTrainingBackend())


def main(argv=None):
    args = shared.parse_args(argv, default_config="cfg/mujoco_latent_finetune.yaml", latent_task_options=True)
    if args.mixed_startup_stage3 and args.three_tasks:
        raise ValueError("--mixed-startup-stage3 and --three-tasks are mutually exclusive")
    if args.config_from_checkpoint and not args.checkpoint:
        raise ValueError("--config-from-checkpoint requires --checkpoint")
    cfg = (shared._config_from_checkpoint(args.checkpoint, LatentPPOConfig) if args.config_from_checkpoint
           else shared._load_config(args.config, LatentPPOConfig))
    if args.three_tasks:
        cfg.three_task_enable = cfg.mixed_task_enable = True
    for key, value in {"checkpoint_path": args.checkpoint, "xml_path": args.xml, "device": args.device,
                       "total_updates": args.updates, "num_envs": args.num_envs, "seed": args.seed,
                       "env_workers": args.env_workers}.items():
        if value is not None:
            setattr(cfg, key, value)
    if args.mixed_startup_stage3:
        configure_startup_stage3_mix(cfg)
    shared._apply_cli_overrides(cfg, args.config_overrides)
    if args.eval_only or cfg.eval_only:
        backend = LatentTrainingBackend()
        backend.validate(cfg)
        policy, metadata = backend.make_policy(cfg, None)
        print("Loaded checkpoint:", metadata)
        shared.run_evaluation_suite(policy, cfg, cfg.dr_curriculum_levels[cfg.dr_curriculum_initial_level])
    else:
        train(cfg)


if __name__ == "__main__":
    main()
