"""Joint policy migration/resume, including the optional encoder extension."""
import torch

from .layout import SRL39Layout
from .runtime import check_checkpoint_mode


def _insert_design_columns(weights, input_keys):
    """Seed new weights and means with zero, and new running variances with one."""
    expanded = weights.copy()
    for key in input_keys:
        if key not in weights:
            continue
        tensor = weights[key]
        # Actor, privileged critic and central critic all end with the same SRL commands.
        columns = tensor.new_full((*tensor.shape[:-1], SRL39Layout.design_dim),
                                  1 if key == 'running_mean_std.running_var' else 0)
        tail = SRL39Layout.command_dim
        expanded[key] = torch.cat((tensor[..., :-tail], columns, tensor[..., -tail:]), dim=-1)
    return expanded


def load_joint_checkpoint(agent, checkpoint, resume=False, set_epoch=True):
    check_checkpoint_mode(agent.hmi_encoder, checkpoint)
    srl_weights = checkpoint['model_srl']
    critic_weights = checkpoint.get('central_critic')
    actor_input = 'a2c_network.actor_mlp.0.weight'
    seed_width = srl_weights[actor_input].shape[-1]
    # Real HRI uses 198 features, the original SRL_HRI model uses 258.
    expand = (seed_width in (SRL39Layout.obs_dim, 258)
              and agent.model_srl.a2c_network.actor_mlp[0].in_features
              == seed_width + SRL39Layout.design_dim)
    if expand:
        if resume:
            raise ValueError('Use hsrl_checkpoint to initialize a morphology-conditioned policy from a fixed-shape checkpoint')
        srl_weights = _insert_design_columns(srl_weights, (
            actor_input, 'a2c_network.critic_mlp.0.weight',
            'running_mean_std.running_mean', 'running_mean_std.running_var'))
        if critic_weights is not None:
            critic_weights = _insert_design_columns(critic_weights, ('trunk.0.weight',))
    agent.model.load_state_dict(checkpoint['model'], strict=True)
    agent.model_srl.load_state_dict(srl_weights, strict=True)
    if agent._normalize_amp_input:
        agent._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
    if agent.use_central_critic and critic_weights is not None:
        agent.central_critic.load_state_dict(critic_weights, strict=True)
    elif agent.use_central_critic and agent.hmi_encoder is not None:
        raise ValueError('Encoder initialization requires a joint stage4 checkpoint with central critic')
    encoder_state = checkpoint.get('hmi_encoder')
    if encoder_state is not None:
        agent.hmi_encoder.load(encoder_state, resume_optimizer=resume)
    if resume:
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.optimizer_srl.load_state_dict(checkpoint['optimizer_srl'])
        if agent.use_central_critic:
            agent.central_critic_opt.load_state_dict(checkpoint['central_critic_opt'])
        if set_epoch:
            agent.epoch_num = checkpoint.get('epoch', 0)
            agent._start_frame = checkpoint.get('frame', 0)
        agent.last_mean_rewards = checkpoint.get('last_mean_rewards', -100500)
        agent.update_counter = checkpoint.get('joint_update_counter', 0)
        agent.last_lr = checkpoint.get('joint_last_lr', agent.optimizer_srl.param_groups[0]['lr'])
        for key, scaler in (('scaler', agent.scaler), ('scaler_srl', agent.scaler_srl),
                            ('central_critic_scaler', getattr(agent, 'central_critic_scaler', None))):
            if scaler is not None and key in checkpoint:
                scaler.load_state_dict(checkpoint[key])
        agent._joint_restored = True
    if agent.hmi_encoder is not None:
        agent.update_lr(agent.last_lr)
        if agent.model_srl.normalize_input:
            agent.model_srl.running_mean_std.eval()
