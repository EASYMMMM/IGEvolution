"""Additional matched-validity history controls and 5-frame linear probes."""
import argparse
import json
from pathlib import Path
import numpy as np
import analyze as a
import evaluate as ev
import torch
from isaacgymenvs.learning.SRLEvo.hmi_encoder.model import HMIEncoder
from isaacgymenvs.learning.SRLEvo.hmi_encoder.config import EncoderConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output.resolve();rng=np.random.default_rng(819)
    torch.set_num_threads(4)
    ds=[a.collect_samples(out/f'intact_seed{s}_data.npz',s) for s in (101,202,303)]
    _,ckpt,(_,robot)=ev.load_policy('cpu')
    encoder=HMIEncoder(EncoderConfig.from_dict(ckpt['hmi_encoder']['config'])).eval()
    encoder.load_state_dict(ckpt['hmi_encoder']['model'])
    names=['intact','shuffle_old59','hold_old59','tile_recent5','last5_zero']
    predictions={n:[] for n in names};actions={n:[] for n in names};ys=[]
    with torch.no_grad():
        for d in ds[1:]:
            features=d['data']['features']
            take=rng.choice(len(d['y']),2000,replace=False)
            for offset in range(0,len(take),128):
                ix=take[offset:offset+128];t,e=d['time'][ix],d['env'][ix]
                h=torch.from_numpy(features[t[:,None]+np.arange(-63,1),e[:,None]].copy())
                obs=torch.from_numpy(d['obs'][ix].copy())
                z,pr=encoder(h);ref=ev.actor(robot,obs,z).clamp(-1,1)
                h_shuffle=h.clone();perm=torch.from_numpy(rng.permutation(59));h_shuffle[:,:59]=h[:,perm]
                h_hold=h.clone();h_hold[:,:59]=h[:,59:60]
                h_tile=h[:,torch.as_tensor(59+(np.arange(64)-59)%5)].clone()
                h_zero=h.clone();h_zero[:,:59]=0
                histories=dict(intact=h,shuffle_old59=h_shuffle,hold_old59=h_hold,tile_recent5=h_tile,last5_zero=h_zero)
                for name,hist in histories.items():
                    zk,pk=encoder(hist);ak=ev.actor(robot,obs,zk).clamp(-1,1)
                    predictions[name].append(pk.numpy().reshape(-1,9))
                    actions[name].append(np.sqrt(((ak-ref).numpy()**2).mean(1)))
                ys.append(d['y'][ix])
    y=np.concatenate(ys);rows=[]
    for name in names:
        sc=a.score(y,np.concatenate(predictions[name]))
        rows.append(dict(condition=name,action_rms=float(np.concatenate(actions[name]).mean()),
                         **{f'rmse_h{h}_{j}':float(sc['rmse'][i*3+j]) for i,h in enumerate((0,8,16)) for j in range(3)},
                         **{f'r2_h{h}_{j}':float(sc['r2'][i*3+j]) for i,h in enumerate((0,8,16)) for j in range(3)}))
    a.write_csv(out/'history_validity_controls.csv',rows)
    raw5=[]
    for d in ds:
        f=d['data']['features'];t,e=d['time'],d['env']
        raw5.append(f[t[:,None]+np.arange(-4,1),e[:,None],:40].reshape(-1,200))
    existing=json.loads((out/'results.json').read_text())
    existing['probes']=[r for r in existing['probes'] if not r['probe'].startswith('Recent 5')]
    for name,xtrain,xtest in [('Recent 5 raw frames (200D)',raw5[0],np.concatenate(raw5[1:])),
                             ('Recent 5 + z (215D)',np.c_[raw5[0],ds[0]['z']],np.concatenate([np.c_[raw5[i],ds[i]['z']] for i in (1,2)]))]:
        train=ds[0];fit=train['env']<48;val=~fit
        best=None
        for reg in (1e-5,1e-4,1e-3,.01,.1,1,10):
            m=a.ridge_fit(xtrain[fit].astype('float64'),train['y'][fit],reg)
            error=np.mean(((a.ridge_predict(m,xtrain[val])-train['y'][val])**2).mean(0)/np.maximum(train['y'][fit].var(0),1e-5))
            if best is None or error<best[0]:best=(error,reg)
        m=a.ridge_fit(xtrain.astype('float64'),train['y'],best[1])
        ytest=np.concatenate([d['y'] for d in ds[1:]])
        sc=a.score(ytest,a.ridge_predict(m,xtest))
        for j in range(9):
            existing['probes'].append(dict(probe=name,ridge_alpha=best[1],horizon_steps=(0,8,16)[j//3],target=a.TARGETS[j%3],r2=float(sc['r2'][j]),rmse=float(sc['rmse'][j])))
    a.write_csv(out/'linear_probe_metrics.csv',existing['probes'])
    existing['history_validity_controls']=rows
    existing['history_validity_samples']=len(y)
    (out/'results.json').write_text(json.dumps(existing,indent=2))
    fig,axes=a.plt.subplots(1,2,figsize=(13,4.5))
    plotlabels=['Intact','Shuffle old 59','Hold old 59','Tile recent 5','Zero old 59']
    for j,target in enumerate(('vx','vy','yaw rate')):
        axes[0].bar(np.arange(5)+j*.24,[r[f'rmse_h0_{j}'] for r in rows],width=.24,label=target)
    axes[0].set_xticks(np.arange(5)+.24);axes[0].set_xticklabels(plotlabels,rotation=20,ha='right')
    axes[0].set(ylabel='RMSE (m/s or rad/s)',title='Human-state estimation');axes[0].legend()
    axes[1].bar(range(5),[r['action_rms'] for r in rows],color='#1f7a8c')
    axes[1].set_xticks(range(5));axes[1].set_xticklabels(plotlabels,rotation=20,ha='right')
    axes[1].set(ylabel='Mean clipped action RMS change',title='Control response, raw Actor observation fixed')
    fig.suptitle('Long-history controls: preserve recent 5 frames; separate validity from temporal content',fontsize=13)
    fig.tight_layout();a.savefig(fig,out/'figures','08_history_controls')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
