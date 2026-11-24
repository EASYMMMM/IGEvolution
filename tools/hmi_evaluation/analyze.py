"""Analyze frozen encoder interventions and render scientific figures/tables.

All regression probes use seed 101 only; hyperparameters use disjoint env IDs.
Seeds 202/303 are held out. Statistical resampling uses episodes, never frames.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import evaluate as ev
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from isaacgymenvs.learning.SRLEvo.hmi_encoder.config import EncoderConfig
from isaacgymenvs.learning.SRLEvo.hmi_encoder.model import HMIEncoder

LABELS = {'intact':'Intact', 'zero':'Zero z', 'mean':'Mean z', 'shuffle':'Swap z',
          'delay16':'Delay 240 ms', 'human_swap':'Swap HMI inputs', 'short5':'Last 5 frames',
          'intact_repeat':'Intact repeat', 'pose_swap':'Swap orientation',
          'force_swap':'Swap force', 'legs_swap':'Swap leg pose',
          'robot_swap':'Swap robot inputs', 'reverse':'Reverse history'}
COLORS = ['#1f7a8c','#e15759','#f28e2b','#9467bd','#4e79a7','#b07aa1','#76b7b2','#8b8b8b']
TARGETS = ['vx (m/s)', 'vy (m/s)', 'yaw rate (rad/s)']


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(y, pred):
    return dict(rmse=np.sqrt(np.mean((y-pred)**2, axis=0)),
                mae=np.mean(np.abs(y-pred), axis=0),
                r2=1-np.sum((y-pred)**2, axis=0)/np.maximum(np.sum((y-y.mean(0))**2, axis=0), 1e-12))


def cluster_boot(values, groups, rng, n=1500):
    unique=np.unique(groups)
    sums=np.asarray([np.sum(values[groups == g],axis=0) for g in unique])
    counts=np.asarray([np.sum(groups == g) for g in unique])
    indices=rng.integers(len(unique),size=(n,len(unique)))
    denominator=counts[indices].sum(1)
    if values.ndim>1:denominator=denominator[:,None]
    samples=sums[indices].sum(1)/denominator
    return np.percentile(samples, [2.5, 97.5], axis=0)


def savefig(fig, out, name):
    fig.savefig(out / (name+'.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(out / (name+'.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def collect_samples(path, seed):
    data = np.load(path)
    states, active, ep = data['states'], data['active'], data['episode']
    times = data['sample_steps']
    # Full 64-frame history and all three target times must belong to first episode.
    valid_t = (times >= 64) & (times + 16 < len(states))
    rows = np.nonzero(valid_t)[0]
    t = times[rows]
    mask = active[t] & active[t+16] & (ep[t-63] == ep[t+16])
    ri, ei = np.nonzero(mask)
    ti, rows = t[ri], rows[ri]
    yaw = states[ti, ei, 3]
    targets = []
    for horizon in (0, 8, 16):
        st = states[ti+horizon, ei]
        dyaw = st[:, 2] - states[ti+horizon-1, ei, 2]
        yaw_rate = np.arctan2(np.sin(dyaw), np.cos(dyaw)) / .015
        targets.append(np.stack((np.cos(yaw)*st[:, 0]+np.sin(yaw)*st[:, 1],
                                 -np.sin(yaw)*st[:, 0]+np.cos(yaw)*st[:, 1], yaw_rate), 1))
    return dict(data=data, seed=seed, time=ti, env=ei, rows=rows,
                z=data['z_all'][ti, ei], obs=data['obs'][rows, ei],
                pred=data['prediction'][rows, ei].reshape(-1, 9),
                y=np.stack(targets, 1).reshape(-1, 9),
                raw=data['features'][ti, ei, :40], group=seed*1000+ei)


def ridge_fit(x, y, reg):
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    xs = (x-mean)/std
    ym = y.mean(0)
    w = np.linalg.solve(xs.T@xs / len(xs) + reg*np.eye(x.shape[1]), xs.T@(y-ym)/len(xs))
    return (mean, std, ym, w)


def ridge_predict(model, x):
    mean, std, ym, w = model
    return ((x-mean)/std)@w + ym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    rng = np.random.default_rng(715)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'axes.titleweight':'bold', 'savefig.dpi':180})
    datasets = [collect_samples(out/f'intact_seed{s}_data.npz', s) for s in (101, 202, 303)]
    train = datasets[0]
    test = {k:np.concatenate([d[k] for d in datasets[1:]]) for k in ('z','obs','pred','y','raw','group')}
    group_test = test['group']
    print('DATA', [(d['seed'], len(d['y'])) for d in datasets], flush=True)

    # 1. Semantic information: frozen prediction head, shuffled-pair negative
    # control, current-estimate persistence, and held-out linear probes.
    frozen = score(test['y'], test['pred'])
    perm = rng.permutation(len(test['pred']))
    shuffled = score(test['y'], test['pred'][perm])
    persistence = np.tile(test['pred'][:, :3], (1, 3))
    persistent = score(test['y'], persistence)
    oracle_persistence = score(test['y'], np.tile(test['y'][:, :3], (1, 3)))
    prediction_rows = []
    pred_models = {'Frozen prediction head':test['pred'], 'Shuffled latent/head':test['pred'][perm],
                   'Current-head persistence':persistence,
                   'True-current persistence (privileged reference)':np.tile(test['y'][:, :3], (1,3))}
    for name, pred in pred_models.items():
        stats = score(test['y'], pred)
        ci = np.sqrt(cluster_boot((test['y']-pred)**2, group_test, rng))
        for j in range(9):
            prediction_rows.append(dict(model=name, horizon_steps=(0,8,16)[j//3], target=TARGETS[j%3],
                                        rmse=stats['rmse'][j], mae=stats['mae'][j], r2=stats['r2'][j],
                                        episode_bootstrap_rmse_ci_low=ci[0,j], episode_bootstrap_rmse_ci_high=ci[1,j]))
    probe_rows, probe_models = [], {}
    probe_inputs = {'z (15D)':(train['z'], test['z']),
                    'Current raw inputs (40D)':(train['raw'], test['raw']),
                    'Raw + z (55D)':(np.c_[train['raw'],train['z']], np.c_[test['raw'],test['z']])}
    fit_mask, val_mask = train['env'] < 48, train['env'] >= 48
    for name, (xtr, xte) in probe_inputs.items():
        errors = []
        grid = (1e-5,1e-4,1e-3,1e-2,.1,1,10)
        for reg in grid:
            model = ridge_fit(xtr[fit_mask].astype('float64'), train['y'][fit_mask], reg)
            err = ((ridge_predict(model,xtr[val_mask])-train['y'][val_mask])**2).mean(0)
            errors.append(np.mean(err / np.maximum(train['y'][fit_mask].var(0),1e-5)))
        reg = grid[int(np.argmin(errors))]
        model = ridge_fit(xtr.astype('float64'),train['y'],reg)
        predictions = ridge_predict(model,xte)
        probe_models[name] = predictions
        for j, r2 in enumerate(score(test['y'],predictions)['r2']):
            probe_rows.append(dict(probe=name, ridge_alpha=reg, horizon_steps=(0,8,16)[j//3],
                                   target=TARGETS[j%3], r2=r2,
                                   rmse=score(test['y'],predictions)['rmse'][j]))
    write_csv(out/'prediction_metrics.csv',prediction_rows)
    write_csv(out/'linear_probe_metrics.csv',probe_rows)
    fig, ax = plt.subplots(1, 3, figsize=(15,4.2), gridspec_kw={'width_ratios':[1,1.25,1.25]})
    im=ax[0].imshow(frozen['r2'].reshape(3,3),vmin=0,vmax=1,cmap='YlGnBu')
    for i in range(3):
        for j in range(3):
            ax[0].text(j,i,f"{frozen['r2'].reshape(3,3)[i,j]:.3f}",ha='center',va='center',color='white' if frozen['r2'].reshape(3,3)[i,j]>.6 else 'black')
    ax[0].set_xticks(range(3));ax[0].set_xticklabels(['vx','vy','yaw rate'])
    ax[0].set_yticks(range(3));ax[0].set_yticklabels(['Now','+120 ms','+240 ms'])
    ax[0].set_title('Frozen head: held-out R²')
    fig.colorbar(im,ax=ax[0],fraction=.046)
    for j, target in enumerate(['vx','vy','yaw rate']):
        color=['#1f77b4','#2ca02c','#9467bd'][j]
        ax[1].plot([0,120,240],frozen['rmse'].reshape(3,3)[:,j],'-o',label=target,color=color)
        ax[1].plot([0,120,240],persistent['rmse'].reshape(3,3)[:,j],'--',alpha=.55,color=color)
    ax[1].set(xlabel='Prediction horizon (ms)',ylabel='RMSE (m/s or rad/s)',title='Solid: prediction; dashed: persistence')
    ax[1].legend()
    for i,(name,pred) in enumerate(probe_models.items()):
        ax[2].bar(np.arange(3)+i*.24,score(test['y'],pred)['r2'][:3],width=.24,label=name)
    ax[2].set_xticks(np.arange(3)+.24);ax[2].set_xticklabels(['vx','vy','yaw rate'])
    ax[2].set(ylabel='Held-out R²',title='Linear readout of current human motion')
    ax[2].legend(fontsize=8,loc='lower center')
    fig.suptitle('Does the latent contain human motion information?',fontsize=16,y=1.04)
    fig.tight_layout();savefig(fig,figures,'01_information')

    fig, axes=plt.subplots(2,3,figsize=(12,7))
    for row, horizon in enumerate((0,16)):
        for j in range(3):
            col=(0,8,16).index(horizon)*3+j
            a=axes[row,j]
            a.hexbin(test['y'][:,col],test['pred'][:,col],gridsize=50,bins='log',cmap='viridis',mincnt=1)
            lo,hi=np.percentile(test['y'][:,col],[.1,99.9]);a.plot([lo,hi],[lo,hi],'r--',lw=1)
            a.set(xlabel='True '+TARGETS[j],ylabel='Predicted',title=f"{horizon*15} ms | R²={frozen['r2'][col]:.3f}")
    fig.suptitle('Frozen prediction head on unseen evaluation trajectories',fontsize=15)
    fig.tight_layout();savefig(fig,figures,'02_prediction_scatter')

    # 2. Keep the original Actor observation fixed and change ONLY the latent.
    _, checkpoint, (_, robot)=ev.load_policy('cpu')
    encoder=HMIEncoder(EncoderConfig.from_dict(checkpoint['hmi_encoder']['config'])).eval()
    encoder.load_state_dict(checkpoint['hmi_encoder']['model'])
    for p in encoder.parameters():p.requires_grad_(False)
    mean_z=np.load(out/'calibration_mean.npy')
    interventions=['zero','mean','shuffle','delay16','human_swap','short5','pose_swap','force_swap','legs_swap','robot_swap','reverse']
    results={k:[] for k in interventions}
    latent_results={k:[] for k in interventions}
    kl_results={k:[] for k in interventions}
    group_ids=[]; all_obs=[];all_z=[];all_actions=[];gradient_histories=[];gradient_observations=[]
    memory_lengths=(1,5,16,32,64)
    memory_errors={k:[] for k in memory_lengths};memory_actions={k:[] for k in memory_lengths}
    sigma=robot.a2c_network.sigma.detach().exp().numpy()
    with torch.no_grad():
        for ds in datasets[1:]:
            print('COUNTERFACTUALS seed',ds['seed'],flush=True)
            take=rng.choice(len(ds['y']),min(6000,len(ds['y'])),replace=False)
            data=ds['data']
            f=data['features'];eps=data['episode']
            for offset in range(0,len(take),256):
                ix=take[offset:offset+256]
                t,e=ds['time'][ix],ds['env'][ix]
                times=t[:,None]+np.arange(-63,1)
                history=torch.from_numpy(f[times,e[:,None]].copy())
                donor_e=(e-1)%f.shape[1]
                donor=torch.from_numpy(f[times,donor_e[:,None]].copy())
                donor*=torch.from_numpy((eps[times,donor_e[:,None]]==eps[t,donor_e][:,None])[...,None])
                obs=torch.from_numpy(ds['obs'][ix].copy())
                z,pred=encoder(history)
                torch.testing.assert_close(z,torch.from_numpy(ds['z'][ix]),atol=2e-5,rtol=2e-5)
                mu=ev.actor(robot,obs,z)
                action=mu.clamp(-1,1)
                zs={'zero':torch.zeros_like(z),'mean':torch.from_numpy(mean_z).expand_as(z),
                    'shuffle':torch.from_numpy(data['z_all'][t,donor_e].copy()),
                    'delay16':torch.from_numpy(data['z_all'][t-16,e].copy())}
                for name,sl in [('human_swap',slice(0,19)),('pose_swap',slice(0,12)),('force_swap',slice(12,15)),
                                ('legs_swap',slice(15,19)),('robot_swap',slice(19,40))]:
                    hist=history.clone();hist[...,sl]=donor[...,sl]
                    zs[name]=encoder(hist)[0]
                hist=history.clone();hist[:,:-5]=0;zs['short5']=encoder(hist)[0]
                zs['reverse']=encoder(history.flip(1))[0]
                for length in memory_lengths:
                    if length==64:
                        zk,pk=z,pred
                    else:
                        limited=history.clone();limited[:,:-length]=0
                        zk,pk=encoder(limited)
                    memory_errors[length].append((pk.numpy().reshape(-1,9)-ds['y'][ix])**2)
                    memory_actions[length].append(((ev.actor(robot,obs,zk).clamp(-1,1)-action).numpy()**2).mean(1))
                if len(gradient_histories)<32:
                    gradient_histories.append(history[:16].numpy().copy())
                    gradient_observations.append(obs[:16].numpy().copy())
                for name in interventions:
                    altered_mu=ev.actor(robot,obs,zs[name]);changed=altered_mu.clamp(-1,1)
                    results[name].append((changed-action).numpy())
                    latent_results[name].append((zs[name]-z).numpy())
                    kl_results[name].append((((altered_mu-mu).numpy()/sigma)**2).sum(1)/2)
                group_ids.append(ds['group'][ix]);all_obs.append(obs.numpy());all_z.append(z.numpy());all_actions.append(action.numpy())
    groups=np.concatenate(group_ids);obs=np.concatenate(all_obs);z=np.concatenate(all_z);actions=np.concatenate(all_actions)
    action_rows=[]
    for name in interventions:
        diff=np.concatenate(results[name]);zd=np.concatenate(latent_results[name]);kl=np.concatenate(kl_results[name])
        per_sample=np.sqrt(np.mean(diff**2,1));ci=cluster_boot(per_sample,groups,rng)
        action_rows.append(dict(intervention=name,n_samples=len(diff),mean_action_rms=per_sample.mean(),
                                action_rms_ci_low=ci[0],action_rms_ci_high=ci[1],
                                mean_latent_rms=np.sqrt((zd**2).mean(1)).mean(),
                                mean_policy_kl=kl.mean(),p95_action_rms=np.percentile(per_sample,95)))
    write_csv(out/'action_interventions.csv',action_rows)
    memory_rows=[]
    for length in memory_lengths:
        rmse=np.sqrt(np.concatenate(memory_errors[length]).mean(0))
        memory_rows.append(dict(history_frames=length,
                               mean_action_rms=np.sqrt(np.concatenate(memory_actions[length])).mean(),
                               **{f'rmse_h{h}_{j}':rmse[k*3+j] for k,h in enumerate((0,8,16)) for j in range(3)}))
    write_csv(out/'history_length_metrics.csv',memory_rows)

    # Local temporal sensitivity through the complete encoder-to-Actor path.
    history=torch.from_numpy(np.concatenate(gradient_histories)).requires_grad_(True)
    history_obs=torch.from_numpy(np.concatenate(gradient_observations))
    latent,forecast=encoder(history)
    history_action=ev.actor(robot,history_obs,latent).clamp(-1,1)
    feature_scale=np.r_[train['raw'].std(0),0]
    temporal=np.zeros(64)
    for joint in range(6):
        grad=torch.autograd.grad(history_action[:,joint].sum(),history,retain_graph=True)[0].detach().numpy()
        temporal+=(np.abs(grad)*feature_scale[None,None,:]).mean(0).sum(1)/6
    temporal_fraction=temporal[::-1]/max(temporal.sum(),1e-15)
    temporal_stats=dict(newest_5_fraction=float(temporal_fraction[:5].sum()),
                        newest_16_fraction=float(temporal_fraction[:16].sum()),
                        newest_32_fraction=float(temporal_fraction[:32].sum()),
                        age_contribution=temporal_fraction.tolist(),n_samples=len(history))
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for j,target in enumerate(('vx','vy','yaw rate')):
        axes[0].plot(memory_lengths,[r[f'rmse_h0_{j}'] for r in memory_rows],'-o',label=target)
    axes[0].set(xlabel='Retained recent frames',ylabel='RMSE (m/s or rad/s)',title='Current human-state prediction');axes[0].legend()
    axes[1].plot(memory_lengths,[r['mean_action_rms'] for r in memory_rows],'-o',color='#1f7a8c')
    axes[1].set(xlabel='Retained recent frames',ylabel='Mean action RMS change',title='Action change from 64-frame encoding')
    axes[2].plot(np.arange(1,65),np.cumsum(temporal_fraction),color='#1f7a8c')
    axes[2].axvline(5,color='grey',ls='--');axes[2].axvline(16,color='grey',ls=':')
    axes[2].set(xlabel='Newest frames included',ylabel='Cumulative local sensitivity',ylim=(0,1.02),title='Encoder → action temporal sensitivity')
    fig.suptitle('Is the long history used? Truncation and local derivative checks',fontsize=15,y=1.04)
    fig.tight_layout();savefig(fig,figures,'07_temporal_information')

    # Per-dimension finite intervention and local derivative both include action clipping.
    sample=rng.choice(len(z),min(1024,len(z)),replace=False)
    ob=torch.from_numpy(obs[sample]);zz=torch.from_numpy(z[sample]).requires_grad_(True)
    aa=ev.actor(robot,ob,zz).clamp(-1,1)
    grads=[]
    for joint in range(6):
        grads.append(torch.autograd.grad(aa[:,joint].sum(),zz,retain_graph=True)[0].detach().numpy())
    gradient=np.stack(grads,1)
    sensitivity=np.mean(np.abs(gradient),0)*train['z'].std(0)[None,:]
    one_dim=[]
    with torch.no_grad():
        base=ev.actor(robot,ob,zz.detach()).clamp(-1,1)
        for dim in range(15):
            ab=zz.detach().clone();ab[:,dim]=float(mean_z[dim])
            one_dim.append(np.sqrt(((ev.actor(robot,ob,ab).clamp(-1,1)-base).numpy()**2).mean(0)))
    one_dim=np.asarray(one_dim)
    write_csv(out/'latent_dimension_sensitivity.csv',
              [dict(latent_dim=i,std=float(train['z'][:,i].std()),
                    **{f'action{j}_rms':one_dim[i,j] for j in range(6)}) for i in range(15)])
    fig,ax=plt.subplots(1,2,figsize=(14,5),gridspec_kw={'width_ratios':[1,1.2]})
    vals=np.array([r['mean_action_rms'] for r in action_rows])
    ci=np.array([[r['action_rms_ci_low'],r['action_rms_ci_high']] for r in action_rows])
    ax[0].barh(np.arange(len(vals)),vals,color='#1f7a8c',xerr=np.maximum(0,np.c_[vals-ci[:,0],ci[:,1]-vals].T),capsize=2)
    ax[0].set_yticks(np.arange(len(vals)));ax[0].set_yticklabels([LABELS[x] for x in interventions]);ax[0].invert_yaxis()
    ax[0].set(xlabel='Mean RMS change in clipped action [-1, 1]',title='Same observation, altered latent')
    im=ax[1].imshow(sensitivity,aspect='auto',cmap='magma')
    ax[1].set_xticks(range(15));ax[1].set_xticklabels(range(1,16));ax[1].set_yticks(range(6));ax[1].set_yticklabels([f'Action {j+1}' for j in range(6)])
    ax[1].set(xlabel='Latent dimension',title='Local sensitivity: mean |da/dz| × std(z)')
    fig.colorbar(im,ax=ax[1],fraction=.035)
    fig.suptitle('Does the controller use the latent? Frozen weights and raw observations',fontsize=15,y=1.02)
    fig.tight_layout();savefig(fig,figures,'03_action_causality')

    # 3. Latent geometry is descriptive, not a claim of disentangled semantics.
    zall=test['z'];center=zall-zall.mean(0)
    cov=np.cov(center,rowvar=False);eigen,basis=np.linalg.eigh(cov);order=np.argsort(eigen)[::-1];eigen=eigen[order];basis=basis[:,order]
    rank=float(eigen.sum()**2/np.square(eigen).sum())
    correlation=np.corrcoef(np.c_[zall,test['y'][:,:3]],rowvar=False)[:15,15:]
    geometry=dict(participation_ratio=rank, std=zall.std(0).tolist(),
                   saturation_fraction=(np.abs(zall)>.95).mean(0).tolist(),
                   eigenvalue_fraction=(eigen/eigen.sum()).tolist(),action_std=float(actions.std(0).mean()))
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    ax[0,0].bar(np.arange(15)+1,zall.std(0),color='#1f7a8c');ax[0,0].set(xlabel='Latent dimension',ylabel='Standard deviation',title=f'Variation; effective rank {rank:.2f} / 15')
    ax[0,1].plot(np.arange(15)+1,np.cumsum(eigen/eigen.sum()),'o-',color='#1f7a8c');ax[0,1].set(xlabel='Principal components',ylabel='Cumulative variance',ylim=(0,1.05),title='Latent covariance spectrum')
    im=ax[1,0].imshow(correlation.T,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1)
    ax[1,0].set_xticks(range(15));ax[1,0].set_xticklabels(range(1,16));ax[1,0].set_yticks(range(3));ax[1,0].set_yticklabels(['vx','vy','yaw rate']);ax[1,0].set(xlabel='Latent dimension',title='Correlation with current human motion')
    fig.colorbar(im,ax=ax[1,0],fraction=.035)
    take=rng.choice(len(center),min(6000,len(center)),replace=False);xy=center[take]@basis[:,:2]
    sc=ax[1,1].scatter(xy[:,0],xy[:,1],c=test['y'][take,2],s=3,cmap='coolwarm',vmin=-.6,vmax=.6,rasterized=True,alpha=.6)
    ax[1,1].set(xlabel='PC1',ylabel='PC2',title='Latent PCA colored by human yaw rate');fig.colorbar(sc,ax=ax[1,1],label='rad/s')
    fig.suptitle('Latent geometry (descriptive, not evidence of disentanglement)',fontsize=15)
    fig.tight_layout();savefig(fig,figures,'04_latent_geometry')

    # 4. Matched closed-loop experiments; first-episode metrics only.
    conditions=list(ev.CONDITIONS)+['intact_repeat']
    metrics={};metas={}
    for condition in conditions:
        arrays=[np.load(out/f'{condition}_seed{s}_metrics.npz') for s in (101,202,303)]
        metrics[condition]={k:np.concatenate([a[k] for a in arrays]) for k in ('failed','duration','position_rmse','velocity_rmse','force_mean','power_mean','action_delta_rms','srl_return','activation_mean')}
        metas[condition]=[json.loads((out/f'{condition}_seed{s}_meta.json').read_text()) for s in (101,202,303)]
    for cond in conditions:
        for i in range(3):
            assert metas[cond][i].get('initial_state_full_batch_commit'), 'Exclude initialization pilot runs'
            assert metas[cond][i].get('randomized_physical_layout'), 'Primary runs require randomized layout'
            assert metas[cond][i]['repeat_first_step_action_max_diff'] < .01
            assert metas[cond][i]['initial_hash']==metas['intact'][i]['initial_hash']
            assert metas[cond][i]['trajectory_hash']==metas['intact'][i]['trajectory_hash']
    n=len(metrics['intact']['failed']);boot=rng.integers(n,size=(4000,n))
    closed_rows=[];per_episode=[]
    for condition in conditions:
        row={'condition':condition,'n_episodes':n}
        for metric,v in metrics[condition].items():
            v=v.astype(float);base=metrics['intact'][metric].astype(float)
            ci=np.percentile(v[boot].mean(1),[2.5,97.5]);delta=v-base
            dci=np.percentile(delta[boot].mean(1),[2.5,97.5])
            row.update({metric:float(v.mean()),metric+'_ci_low':float(ci[0]),metric+'_ci_high':float(ci[1]),
                        metric+'_delta':float(delta.mean()),metric+'_delta_ci_low':float(dci[0]),metric+'_delta_ci_high':float(dci[1])})
        closed_rows.append(row)
        for i in range(n):
            per_episode.append(dict(condition=condition,seed=(101,202,303)[i//64],env_id=i%64,
                                    **{k:float(v[i]) for k,v in metrics[condition].items()}))
    write_csv(out/'closed_loop_summary.csv',closed_rows)
    write_csv(out/'closed_loop_episodes.csv',per_episode)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    names=[LABELS[c] for c in conditions]
    for ax,metric,title,scale in zip(axes.flat,['failed','duration','position_rmse','force_mean'],
                                    ['Early termination rate (%)','First-episode duration (s)',
                                     'Position tracking RMSE while alive (m)','Mean interaction force while alive (N)'],[100,1,1,1]):
        vals=np.array([r[metric] for r in closed_rows])*scale
        low=np.array([r[metric+'_ci_low'] for r in closed_rows])*scale
        high=np.array([r[metric+'_ci_high'] for r in closed_rows])*scale
        ax.bar(np.arange(8),vals,color=COLORS,yerr=np.maximum(0,np.array([vals-low,high-vals])),capsize=3)
        ax.set_xticks(range(8));ax.set_xticklabels(names,rotation=28,ha='right',fontsize=8)
        ax.set_title(title)
    fig.suptitle('Closed-loop causal interventions | 192 matched episodes per condition | 75 s cap',fontsize=15)
    fig.tight_layout();savefig(fig,figures,'05_closed_loop')
    fig,ax=plt.subplots(figsize=(10,5))
    for i,condition in enumerate(conditions):
        arrays=[np.load(out/f'{condition}_seed{s}_metrics.npz') for s in (101,202,303)]
        ax.plot(arrays[0]['trace_time'],np.mean([a['trace_survival'] for a in arrays],axis=0),
                label=LABELS[condition],color=COLORS[i],ls='--' if condition=='intact_repeat' else '-')
    ax.set(xlabel='Time (s)',ylabel='Fraction without early termination',ylim=(0,1.02),title='Survival under latent interventions')
    ax.legend(ncol=2);fig.tight_layout();savefig(fig,figures,'06_survival')

    summary=dict(checkpoint=str(ev.CKPT),checkpoint_sha256=hashlib.sha256(ev.CKPT.read_bytes()).hexdigest(),
                 checkpoint_epoch=int(checkpoint['epoch']),seeds=[101,202,303],num_envs_per_condition=64,
                 inference='deterministic mean, clipped to [-1, 1], original action filter',
                 calibration_seed=17,held_out_seeds=[202,303],sample_counts={str(d['seed']):len(d['y']) for d in datasets},
                 paired_initial_and_trajectory_hashes_match=True,
                 randomized_physical_layout=True,initial_state_full_batch_commit=True,
                 repeat_first_step_action_max_diff=[m['repeat_first_step_action_max_diff'] for m in metas['intact']],
                 evaluator_source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                           for name in ('evaluate.py','analyze.py','run_suite.sh')},
                 prediction=prediction_rows,probes=probe_rows,actions=action_rows,geometry=geometry,closed_loop=closed_rows,
                 history=memory_rows,temporal=temporal_stats,
                 caveats=['Single fixed checkpoint and original task distribution.',
                          'Bootstrap resamples episodes; only three generator seeds.',
                          'Failure includes falling/contact/height and path deviation, not only falls.',
                          'Tracking/force/power are averaged until termination and are survival-conditioned.',
                          'Zero, truncated history and mixed feature interventions may be out of distribution.',
                          'All raw Actor HMI features remain unchanged for latent-only interventions.',
                          'Intact repeat estimates numerical variability across separate PhysX environments.'])
    (out/'results.json').write_text(json.dumps(summary,indent=2,
                                             default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
    print('ANALYSIS COMPLETE',json.dumps({'geometry':geometry,'closed_loop':[{k:r[k] for k in ('condition','failed','duration','position_rmse')} for r in closed_rows]}),flush=True)


if __name__=='__main__':main()
