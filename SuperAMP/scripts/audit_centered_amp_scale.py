"""CPU-only scalar reward audit on frozen real trajectories; no policy update."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('frozen_amp_diagnostics',ROOT/'scripts/analyze_amp_recovery_style.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def desc(x):
    x=np.asarray(x,dtype=np.float64)
    return dict(count=int(x.size),mean=float(x.mean()),min=float(x.min()),max=float(x.max()),p05=float(np.quantile(x,.05)),median=float(np.median(x)),p95=float(np.quantile(x,.95)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path('/home/autel/AMP_mjlab_wmac'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'docs/validation/centered_scale_audit.json')
    args=p.parse_args();torch.set_num_threads(2)
    checkpoint=args.checkpoint or args.source_root/'logs/wmac/2026-11-16_11-31-06_022422_seed42_full_official_flat_mirrored/model_40800.pt'
    scorer=mod.load_scorer(checkpoint)
    adapter=mod.make_cpu_adapter(ROOT/'src/assets/robots/unitree_g1/xmls/g1.xml')
    pools={};sources={}
    for mode in ('walk','recovery'):
        directory=args.source_root/f'logs/ppo60000_vs_wmac40800_{mode}_seed2026'
        for role,policy in (('a','ppo60000'),('b','wm40800')):
            path=directory/f'transitions_{role}.npz'
            with np.load(path,allow_pickle=False) as f:
                state=np.asarray(f['state']);nxt=np.asarray(f['next_state'])
                mask=np.asarray(f['valid']).astype(bool)&~np.asarray(f['boundary']).astype(bool)
                recovery_phase=np.asarray(f['recovery_phase']).astype(bool)
            a,b=mod.features(adapter,state),mod.features(adapter,nxt)
            pair=np.concatenate((a,b),-1)
            scores,_=mod.score_pairs(pair.reshape(-1,406),scorer);scores=scores.reshape(mask.shape)
            time=np.arange(mask.shape[0])[:,None]
            selection=mask if mode=='walk' else mask&(time>=5)&(time<100)
            key=f'{policy}_{mode}'
            pools[key]=scores[selection]
            if mode=='recovery':
                pools[f'{policy}_recovery_active']=scores[mask&recovery_phase]
                sources[f'{policy}_recovery_active']={'path':str(path),'sha256':mod.sha256(path),'mask':'valid (first episode) AND NOT boundary AND recovery_phase, exactly (state.height<.55 OR context.delay_clock>0)'}
            sources[key]={'path':str(path),'sha256':mod.sha256(path),'mask':'valid (first episode) AND NOT boundary; recovery only t in [5,100) at 50Hz; walk t in [0,250)'}
    center=.5*(pools['wm40800_walk'].mean()+pools['wm40800_recovery_active'].mean())
    out=dict(schema='instantamp-centered-lsgan-offline-scalar-audit-v1',device='cpu',training_updates=0,physical_steps=0,
             checkpoint=scorer[-1],sources=sources,
             source_reference={'repo':'RoboticsResearchHubVN/InstantAMP','commit':'9fc8031055a18d58e84d9de90639660e7aa6b491','function':'instantamp/amp.py:106-119 style(scale>0)','ema':.99},
             center={'value':float(center),'kind':'offline phase-balanced proxy, NOT a fitted live EMA','construction':'0.5*mean(WMAC real walk valid) + 0.5*mean(WMAC real replay_recovery_phase valid)','limitation':'Replay sample_starts recovery mask and whole training distribution differ; a real training EMA must reuse actual policy_score batches, not this number.'},
             scales={'chosen':1.,'basis':'Half-span of unchanged LSGAN target labels: (1 - (-1))/2 = 1. This is a fixed score-unit adaptation, not an imported WGAN scale or demonstrated optimum.','candidates':[.5,1.,2.,5.]},
             notes=['Frozen recorded inputs and discriminator; scalar chain derivatives only, not policy/state gradient quality or physical behavior.','One shared walk/get-up discriminator; centering mixture may leave a subset in a tanh tail.','tanh can still numerically saturate; no strictly positive gradient guarantee.','Original clipped quadratic reverses sign above score1; centered tanh is monotone and may encourage imagined high-score exploitation.','Mapping changes actual Actor and Critic reward targets; world-model labels/loss/inputs remain independent.'],
             phases={})
    coefficient=scorer[2]
    for key,score in pools.items():
        old_reward=coefficient*np.maximum(1-.25*(score-1)**2,0)
        old_grad=np.where((score>-1)&(score<3),.5*coefficient*(1-score),0)
        item={'score':desc(score),'legacy':{'reward':desc(old_reward),'abs_dr_dd':desc(np.abs(old_grad)),'zero_slope_fraction':float(np.mean(old_grad==0)),'score_above_one_fraction':float(np.mean(score>1))},'centered':{}}
        for scale in (.5,1.,2.,5.):
            z=(score.astype(np.float32)-np.float32(center))/np.float32(scale)
            t=np.tanh(z);rew=np.float32(coefficient*.5)*(1+t)
            grad=np.float32(coefficient/(2*scale))*(1-t*t)
            item['centered'][str(scale)]={'reward':desc(rew),'abs_dr_dd':desc(grad),'zero_slope_fraction':float(np.mean(grad==0)),
                                         'below_one_percent_center_slope_fraction':float(np.mean((1-t*t)<.01)),
                                         'legacy_zero_now_positive_slope_fraction':float(np.mean((old_grad==0)&(grad>0))),
                                         'legacy_zero_count':int(np.sum(old_grad==0)),
                                         'legacy_zero_rescued_count':int(np.sum((old_grad==0)&(grad>0)))}
        out['phases'][key]=item
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'center':float(center),'summary':{k:{'old_zero':v['legacy']['zero_slope_fraction'],'new_zero':v['centered']['1.0']['zero_slope_fraction'],'new_low_slope':v['centered']['1.0']['below_one_percent_center_slope_fraction']}for k,v in out['phases'].items()}},indent=2))
if __name__=='__main__':main()
