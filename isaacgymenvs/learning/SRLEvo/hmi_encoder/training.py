"""Encoder-enabled PPO update; the original SRL update remains untouched.

Microbatches accumulate gradients into ONE optimizer step per PPO minibatch.
Auxiliary losses are normalized by full-minibatch valid counts per horizon,
so reset patterns and microbatch_size do not change their weighting.
"""
import torch
from torch.nn import functional as F
from torch import nn
from rl_games.algos_torch import torch_ext


def update_srl(agent, batch):
    runtime = agent.hmi_encoder
    config = runtime.config
    model = agent.model_srl
    model.train()
    if model.normalize_input:
        model.running_mean_std.eval()  # preserve the complete stage4 normalization
    runtime.model.train()
    agent.optimizer_srl.zero_grad(set_to_none=True)
    runtime.optimizer.zero_grad(set_to_none=True)
    count = batch['obs'].shape[0]
    ids = batch['hmi_sample_ids'].long().flatten()
    env_ids, times = ids // runtime.horizon, ids % runtime.horizon
    denominator = runtime.target_mask[times, env_ids].sum(0).clamp_min(1) * 3
    scale = torch.tensor(config.target_scale, device=runtime.device)
    weights = torch.tensor(config.horizon_weights, device=runtime.device)
    stats = {}

    def record(key, value, weight=1.0):
        value = torch.as_tensor(value, device=runtime.device).detach().mean() * weight
        stats[key] = stats.get(key, 0.0) + value

    for start in range(0, count, config.microbatch_size):
        end = min(count, start + config.microbatch_size)
        sub = {key: value[start:end] if torch.is_tensor(value) else value for key, value in batch.items()}
        fraction = (end - start) / count
        history, targets, valid = runtime.sample(sub['hmi_sample_ids'])
        with torch.cuda.amp.autocast(enabled=agent.mixed_precision):
            payload, predictions = runtime.payload(history)
            result = model({'is_train': True, 'prev_actions': sub['actions'],
                            'obs': agent._preproc_obs(sub['obs']), 'priv_obs': sub['priv_obses'],
                            'actor_latent': payload})
            a_info = agent._actor_loss(sub['old_logp_actions'], result['prev_neglogp'],
                                      sub['advantages'], agent.e_clip)
            actor_loss = a_info.pop('actor_loss').mean()
            c_info = agent._critic_loss(sub['old_values'], result['values'], agent.e_clip,
                                       sub['returns'], agent.clip_value)
            critic_loss = c_info.pop('critic_loss').mean()
            entropy = result['entropy'].mean()
            bounds = agent.bound_loss(result['mus']).mean()
            symmetry = result['mus'].sum() * 0
            error = F.smooth_l1_loss(predictions / scale, targets / scale, reduction='none')
            aux = (error * valid[..., None]).sum((0, 2)) / denominator
            if agent.mirror_loss:
                mirrored_history, mirrored_targets, mirrored_valid = runtime.sample(sub['hmi_sample_ids'], mirrored=True)
                mirrored_payload, mirrored_predictions = runtime.payload(mirrored_history)
                mirrored_result = model({'is_train': False, 'obs': agent._preproc_obs(sub['obs_mirrored']),
                                         'priv_obs': sub['priv_obses_mirrored'], 'actor_latent': mirrored_payload})
                symmetry = agent.sym_loss(result['mus'], mirrored_result['mus'])['sym_loss'].mean()
                mirrored_error = F.smooth_l1_loss(mirrored_predictions / scale, mirrored_targets / scale, reduction='none')
                aux = 0.5 * (aux + (mirrored_error * mirrored_valid[..., None]).sum((0, 2)) / denominator)
            ppo_loss = (actor_loss + agent.critic_coef * critic_loss - agent.entropy_coef * entropy
                        + agent.bounds_loss_coef * bounds + agent.sym_actor_loss_coef * symmetry)
            loss = fraction * ppo_loss + config.auxiliary_coef * (weights * aux).sum()
        agent.scaler_srl.scale(loss).backward()
        record('actor_loss_srl', actor_loss, fraction)
        record('critic_loss_srl', critic_loss, fraction)
        record('entropy_srl', entropy, fraction)
        record('b_loss_srl', bounds, fraction)
        record('sym_loss', symmetry, fraction)
        record('kl_srl', torch_ext.policy_kl(result['mus'].detach(), result['sigmas'].detach(),
                                           sub['mu'], sub['sigma'], reduce=True), fraction)
        for key, value in {**a_info, **c_info}.items():
            record(key, value, fraction)
        for index, horizon in enumerate(config.horizons):
            record(f'hmi_loss_{horizon}', aux[index])

    agent.scaler_srl.unscale_(agent.optimizer_srl)
    agent.scaler_srl.unscale_(runtime.optimizer)
    if agent.truncate_grads:
        nn.utils.clip_grad_norm_(model.parameters(), agent.grad_norm)
        nn.utils.clip_grad_norm_(runtime.model.parameters(), agent.grad_norm)
    if agent._train_srl:
        agent.scaler_srl.step(agent.optimizer_srl)
        agent.scaler_srl.step(runtime.optimizer)
    agent.scaler_srl.update()
    for key, value in dict(last_lr_srl=agent.last_lr, lr_mul_srl=1.0, dagger_loss=0.0, sym_value_loss=0.0, hmi_alpha=runtime.alpha).items():
        record(key, value)
    agent.train_result.update(stats)
