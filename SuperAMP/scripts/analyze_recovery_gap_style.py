"""Read-only common-judge AMP/style and real Bellman diagnostics on paired traces."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from scripts.analyze_amp_recovery_style import (
    make_cpu_adapter, features, expert_reference, load_scorer, sha256, summary, json_clean)
from src.wmac.networks import Critic
from src.wmac.state import quat_to_matrix


def ppo_scorer(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state, old_norm = payload['discriminator_state_dict'], payload['amp_normalizer']
    layers, width = [], 390
    for hidden in (1024, 512, 256):
        layers.extend((torch.nn.Linear(width, hidden),torch.nn.ReLU()))
        width = hidden
    network = torch.nn.Sequential(OrderedDict(trunk=torch.nn.Sequential(*layers),
        amp_linear=torch.nn.Linear(width,1))).eval().requires_grad_(False)
    network.load_state_dict(state)
    mean = torch.tensor(old_norm.mean,dtype=torch.float32)
    std = torch.sqrt(torch.tensor(old_norm.var+old_norm.epsilon,dtype=torch.float32))
    def normalize(x):
        return ((x-mean)/std).clamp(-old_norm.clip_obs,old_norm.clip_obs)
    info = dict(checkpoint=str(path),sha256=sha256(path),feature_dim=195,
        coefficient=.1,coefficient_source='Original repository factory; checkpoint has no saved training configuration',
        epsilon=float(old_norm.epsilon),clip=float(old_norm.clip_obs),
        interpretation='One frozen PPO discriminator scores all policies using the shared legacy195 prefix')
    return network, normalize, .1, info


@torch.no_grad()
def scores(pair, scorer):
    network,norm,coef,info = scorer
    dim = info.get('feature_dim',203)
    values=[]
    for start in range(0,len(pair),1024):
        chunk=torch.as_tensor(pair[start:start+1024],dtype=torch.float32)
        x=torch.cat((norm(chunk[:,:dim]),norm(chunk[:,203:203+dim])),-1)
        values.append(network(x).squeeze(-1).numpy())
    raw=np.concatenate(values)
    return raw,coef*np.maximum(1-.25*(raw-1)**2,0)


@torch.no_grad()
def nearest(query, reference):
    ref=torch.as_tensor(reference,dtype=torch.float64)
    norm=ref.square().sum(-1)
    result=[]
    for start in range(0,len(query),128):
        q=torch.as_tensor(query[start:start+128],dtype=torch.float64)
        squared=(q.square().sum(-1,keepdim=True)+norm[None]-2*q@ref.T).clamp_min(0)
        result.append((squared.min(-1).values/q.shape[-1]).sqrt().numpy())
    return np.concatenate(result)


def masks(arrays,common):
    t=np.arange(arrays['state'].shape[0])[:,None]*.02
    state=arrays['state']
    up=quat_to_matrix(torch.from_numpy(state[...,3:7])).numpy()[...,2,2]
    recovery=(state[...,2]<.55)|(arrays['context'][...,34]>0)
    return {'all':common,'fixed_0p1_to_2':common&(t>=.1)&(t<2),
        'fixed_2_to_5':common&(t>=2)&(t<5),
        'replay_recovery_phase':common&recovery,'replay_walk_phase':common&~recovery,
        'low_or_tilted':common&((state[...,2]<.55)|(up<.5))}


def load_critic(path):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    cfg=payload['config']
    critic=Critic(136,hidden=tuple(cfg['critic_hidden']),activation='gelu',layer_norm=True).eval()
    critic.load_state_dict(payload['critic']);critic.requires_grad_(False)
    return critic,cfg


@torch.no_grad()
def critic_values(critic,adapter,arrays):
    answer=[]
    for successor in (False,True):
        state=arrays['next_state' if successor else 'state'].reshape(-1,71)
        context=arrays['next_context' if successor else 'context'].reshape(-1,35)
        action=arrays['action' if successor else 'previous_action'].reshape(-1,29)
        command=arrays['command'].reshape(-1,3)
        values=[]
        for start in range(0,len(state),2048):
            chunks=[torch.from_numpy(x[start:start+2048]) for x in (state,action,command,context)]
            values.append(critic(adapter.critic_features(*chunks)).numpy())
        answer.append(np.concatenate(values).reshape(arrays['reward'].shape))
    return answer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paired',type=Path,nargs='+',required=True)
    p.add_argument('--wm-checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--ppo',type=Path,default=Path('/home/autel/amp_60000.pt'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-queries',type=int,default=512)
    args=p.parse_args()
    if args.output.exists():p.error('Output exists')
    torch.set_num_threads(2)
    adapter=make_cpu_adapter(ROOT/'src/assets/robots/unitree_g1/xmls/g1.xml')
    mean,std,pools,expert_info=expert_reference(adapter,ROOT/'src/assets/motions/g1/amp',4096,2026)
    scorers={'ppo60000':ppo_scorer(args.ppo)}
    critics={}
    for checkpoint in args.wm_checkpoints:
        key=f'wm{checkpoint.stem.removeprefix("model_")}'
        scorers[key]=load_scorer(checkpoint)
        critics[key]=load_critic(checkpoint)
    report={'schema':'amp-recovery-gap-style-v1','training_updates':0,'device':'cpu',
        'scorers':{key:scorer[3] for key,scorer in scorers.items()},'experts':expert_info,
        'expert_scores':{},'recordings':{},'limitations':[
            'Frozen discriminators are training judges, not independent ground truth. Compare policies within each fixed judge, not score magnitudes between judges.',
            'Nearest distance is over one 20ms pair and cannot prove reference temporal order or correct physical contacts.',
            'Expert motion files contain no policy action or measured contact forces.',
            'Real Bellman residual is an evaluation of existing V; qacc counterfactuals only change numeric rewards, not gradients or policy trajectories.',
            'Primary comparison uses common first-episode fixed windows. Conditional phases may have different occupancy.',
            'Old PPO coefficient is reconstructed from repository defaults; the old checkpoint does not save its full run flags.']}
    for group,pool in pools.items():
        raw=(pool*std+mean).astype(np.float32)
        report['expert_scores'][group]={}
        for key,scorer in scorers.items():
            d,r=scores(raw,scorer)
            report['expert_scores'][group][key]={'raw':summary(d),'style':summary(r),
                'zero_style_fraction':float((r==0).mean())}
    for directory in args.paired:
        metadata=json.loads((directory/'metrics.json').read_text())
        trial_names=list(metadata['trials'])
        data={}
        for trial in trial_names:
            with np.load(directory/f'transitions_{trial}.npz') as z:
                data[trial]={key:np.array(z[key]) for key in ('state','next_state','context','next_context',
                    'previous_action','action','command','reward','terminated','valid','qacc_cost')}
        common=np.logical_and.reduce([value['valid'].astype(bool) for value in data.values()])
        entries={}
        for trial,arrays in data.items():
            print(f'Style diagnosis {directory.name}/{trial}',flush=True)
            pair=np.concatenate((features(adapter,arrays['state']),features(adapter,arrays['next_state'])),-1)
            flat=pair.reshape(-1,406)
            judges={key:scores(flat,scorer) for key,scorer in scorers.items()}
            value_maps={}
            for key,(critic,cfg) in critics.items():
                v,nv=critic_values(critic,adapter,arrays)
                style=judges[key][1].reshape(common.shape)
                reward=cfg['task_weight']*arrays['reward']+cfg['style_weight']*style
                fd=np.square((arrays['next_state'][...,42:71]-arrays['state'][...,42:71])/.02).sum(-1)
                correction=cfg['task_weight']*(-2.5e-7)*.02*(fd-arrays['qacc_cost'])
                residual=reward+cfg['gamma']*(~arrays['terminated'].astype(bool))*nv-v
                value_maps[key]={'value':v,'real_td_residual':residual,'physical_fd_counterfactual_td_residual':residual+correction,
                    'reward':reward,'qacc_definition_correction':correction}
            row={'policy_sha256':metadata['checkpoints'][trial.removesuffix('_repeat')]['sha256'],'phases':{}}
            for phase_index,(phase,mask) in enumerate(masks(arrays,common).items()):
                rng=np.random.default_rng(2026+phase_index)
                indices=np.flatnonzero(mask.reshape(-1))
                selected=indices if len(indices)<=args.max_queries else np.sort(rng.choice(indices,args.max_queries,replace=False))
                normalized=((flat[selected]-mean)/std).astype(np.float32)
                phase_row={'transitions':len(indices),'nn_queries':len(selected),'judge':{},'critic':{}}
                if len(selected):
                    phase_row['expert_pair_rmse']={group:summary(nearest(normalized,pool)) for group,pool in pools.items()}
                for key,(d,r) in judges.items():
                    raw,rew=d.reshape(common.shape)[mask],r.reshape(common.shape)[mask]
                    phase_row['judge'][key]={'raw_score':summary(raw),'style_reward':summary(rew),
                        'zero_gradient_fraction':float(((raw<=-1)|(raw>=3)).mean()) if len(raw) else None}
                for key,mapping in value_maps.items():
                    phase_row['critic'][key]={name:summary(value[mask]) for name,value in mapping.items()}
                row['phases'][phase]=phase_row
            entries[trial]=row
        report['recordings'][directory.name]={'protocol':metadata['protocol'],
            'metrics_sha256':sha256(directory/'metrics.json'),'trials':entries}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(json_clean(report),indent=2,ensure_ascii=False)+'\n')
    print(args.output)


if __name__=='__main__':main()
