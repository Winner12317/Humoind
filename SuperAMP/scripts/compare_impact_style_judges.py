"""Frozen common-judge style scores for paired real physical traces; no training."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.analyze_amp_recovery_style import make_cpu_adapter,features,load_scorer,sha256,json_clean
from scripts.analyze_recovery_gap_style import ppo_scorer,scores
from src.wmac.amp_rewards import CenteredAmpReward


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--ppo',type=Path,default=Path('/home/autel/amp_60000.pt'))
    parser.add_argument('--paired',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);started=time.perf_counter()
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=payload['config']
    if cfg['amp_reward_mapping']!='instantamp_centered_tanh_v1':raise ValueError('Expected centered mapping')
    mapper=CenteredAmpReward(cfg['amp_reward_coef'],cfg['amp_score_scale'],cfg['amp_score_ema_decay'])
    CenteredAmpReward.validate_state(payload['amp_reward_state']);mapper.load_state_dict(payload['amp_reward_state'])
    judges={'current96800':load_scorer(args.checkpoint),'ppo60000':ppo_scorer(args.ppo)}
    adapter=make_cpu_adapter(ROOT/'src/assets/robots/unitree_g1/xmls/g1.xml')
    report={'schema':'impact-common-real-style-judges-v1','device':'cpu','training_updates':0,'physical_steps':0,
            'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha256(args.checkpoint),
            'robot_score_ema':float(mapper.robot_mean),'scale':mapper.scale,
            'judges':{key:value[3] for key,value in judges.items()},'recordings':{},'aggregate_fixed_windows':{},
            'limitations':['Each judge is frozen and compares both policies on actual physics pairs. Neither is independent style ground truth.',
                           'Raw scores or reward magnitudes from different judges must not be compared.',
                           'Quad is a mapping-only counterfactual on the same current D scores, not a retrained policy experiment.',
                           'Aggregates first average repeated rollouts within start, then equal-weight all starts; frames are not independent samples.',
                           'Current judge values cannot establish reference temporal sequence or perceived naturalness.',
                           'Old PPO coefficient .1 is reconstructed from repository defaults; checkpoint lacks full run config.']}
    pooled={}
    for directory in args.paired:
        rows={}
        for trial in ['a','a_repeat','b','b_repeat']:
            with np.load(directory/f'transitions_{trial}.npz') as z:
                rows[trial]={key:np.array(z[key]) for key in ['state','next_state','context','valid','boundary','recovery_phase','reward']}
        common=np.logical_and.reduce([row['valid']&~row['boundary'] for row in rows.values()])
        time_axis=np.arange(common.shape[0])[:,None]*.02
        common_phases={'fixed_0p1_to_2':common&(time_axis>=.1)&(time_axis<2),
                       'fixed_2_to_5':common&(time_axis>=2)&(time_axis<5),'all':common}
        all_recovery=np.logical_and.reduce([row['recovery_phase'] for row in rows.values()])
        all_nonrecovery=np.logical_and.reduce([~row['recovery_phase'] for row in rows.values()])
        common_phases['shared_recovery_fixed_0p1_to_2']=common_phases['fixed_0p1_to_2']&all_recovery
        common_phases['shared_nonrecovery_fixed_0p1_to_2']=common_phases['fixed_0p1_to_2']&all_nonrecovery
        entries={};means={}
        for trial,row in rows.items():
            pair=np.concatenate((features(adapter,row['state']),features(adapter,row['next_state'])),-1).reshape(-1,406)
            d_current,quad=scores(pair,judges['current96800'])
            d_ppo,ppo_quad=scores(pair,judges['ppo60000'])
            with torch.no_grad():
                score=torch.from_numpy(d_current);tanh=mapper(score).numpy();slope=mapper.score_slope(score).numpy()
            maps={'current_raw_score':d_current,'current_tanh_style':tanh,'current_quad_counterfactual_style':quad,
                  'current_tanh_zero_slope':(slope==0).astype(float),'current_quad_zero_slope':((d_current<-1)|(d_current>3)).astype(float),
                  'ppo_judge_raw_score':d_ppo,'ppo_judge_quad_style':ppo_quad}
            maps={key:value.reshape(common.shape) for key,value in maps.items()}
            maps['physical_task_reward']=row['reward']
            maps['task_weighted']=cfg['task_weight']*row['reward']
            maps['current_tanh_style_weighted']=cfg['style_weight']*maps['current_tanh_style']
            maps['combined_current_common_objective']=maps['task_weighted']+maps['current_tanh_style_weighted']
            maps['combined_quad_mapping_counterfactual']=maps['task_weighted']+cfg['style_weight']*maps['current_quad_counterfactual_style']
            phase_masks={**common_phases,'own_recovery_active':common&row['recovery_phase'],'own_walk_phase':common&~row['recovery_phase']}
            entries[trial]={'data_sha256':sha256(directory/f'transitions_{trial}.npz'),'phases':{}}
            means[trial]={}
            for phase,mask in phase_masks.items():
                entries[trial]['phases'][phase]={'transitions':int(mask.sum()),**{key:float(value[mask].mean()) if mask.any() else None for key,value in maps.items()}}
                if phase in common_phases:
                    denominator=mask.sum(0)
                    means[trial][phase]={key:np.sum(np.where(mask,value,0),axis=0)/np.where(denominator>0,denominator,np.nan) for key,value in maps.items()}
            print(f'completed {directory.name}/{trial}',flush=True)
        for phase in common_phases:
            for policy in ['a','b']:
                dst=pooled.setdefault(phase,{}).setdefault(policy,{})
                for key in means[policy][phase]:
                    values=.5*(means[policy][phase][key]+means[policy+'_repeat'][phase][key])
                    dst.setdefault(key,[]).extend(values.tolist())
        report['recordings'][directory.name]={'metrics_sha256':sha256(directory/'metrics.json'),'trials':entries}
    for phase,policies in pooled.items():
        report['aggregate_fixed_windows'][phase]={policy:{key:{'mean':float(np.nanmean(values)) if np.isfinite(values).any() else None,'per_start_repeats_averaged':[v if np.isfinite(v) else None for v in values],'starts':int(np.isfinite(values).sum()),'starts_total':len(values)} for key,values in maps.items()} for policy,maps in policies.items()}
    report['wall_seconds']=time.perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(json_clean(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
