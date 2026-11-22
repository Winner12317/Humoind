"""Read-only CPU H1 model / exact centered-AMP diagnostics on recorded physics.

No simulator, optimizer step, calibration, or change to the model is made.
Kinematic trapezoid residuals are diagnostics, not a physical constraint proof.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.wmac.networks import WorldModel
from src.wmac.amp_rewards import CenteredAmpReward
from scripts.analyze_amp_recovery_style import make_cpu_adapter, load_scorer


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def stats(value):
    x=np.asarray(value,dtype=np.float64).reshape(-1)
    x=x[np.isfinite(x)]
    if not len(x): return {'count':0}
    return dict(count=len(x),mean=float(x.mean()),rms=float(np.sqrt(np.mean(x*x))),
                p05=float(np.quantile(x,.05)),p50=float(np.quantile(x,.5)),p95=float(np.quantile(x,.95)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=128)
    args=parser.parse_args(); torch.set_num_threads(4); started=time.perf_counter()
    p=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=p['config']
    if cfg.get('amp_reward_mapping')!='instantamp_centered_tanh_v1':raise ValueError('Requires centered checkpoint')
    model=WorldModel(hidden=cfg['model_hidden'],activation='gelu',output_zero_init=False).eval()
    model.load_state_dict(p['world_model']);model.requires_grad_(False)
    network,norm,coefficient,info=load_scorer(args.checkpoint)
    CenteredAmpReward.validate_state(p['amp_reward_state'])
    mapper=CenteredAmpReward(coefficient,cfg['amp_score_scale'],cfg['amp_score_ema_decay'])
    mapper.load_state_dict(p['amp_reward_state'])
    adapter=make_cpu_adapter(ROOT/'src/assets/robots/unitree_g1/xmls/g1.xml')
    result={'schema':'impact-frozen-h1-kinematic-style-diagnostic-v1','checkpoint':str(args.checkpoint),'checkpoint_sha256':sha(args.checkpoint),'recording':str(args.recording),
            'cpu_only':True,'training_updates':0,'physical_steps':0,'control_dt':.02,'horizon':1,
            'robot_score_ema':float(mapper.robot_mean),'score_scale':mapper.scale,
            'limitations':['One frozen checkpoint and first paired trial only. No training intervention or causal attribution.',
                           'Phase-conditioned means use each policy own trajectory; they are not matched fixed-phase counterfactuals.',
                           'Trapezoid velocity integration is an approximate consistency diagnostic even for true semi-implicit contact physics.',
                           'H1 teacher-forced predictions do not test H32 open-loop derivatives or full Actor objective.'], 'policies':{}}
    first={}
    for name in ['a','b']:
        with np.load(args.recording/f'transitions_{name}.npz') as f:
            first[name]={k:np.array(f[k],copy=True) for k in ['state','next_state','context','action','valid','boundary','recovery_phase','qacc_cost']}
    common=(first['a']['valid'] & first['b']['valid'] & ~first['a']['boundary'] & ~first['b']['boundary'])
    window=((np.arange(common.shape[0])*.02>=.1)&(np.arange(common.shape[0])*.02<2))[:,None]
    for name,data in first.items():
        shape=data['valid'].shape
        states=torch.from_numpy(data['state'].reshape(-1,71))
        target=torch.from_numpy(data['next_state'].reshape(-1,71))
        actions=torch.from_numpy(data['action'].reshape(-1,29))
        contexts=torch.from_numpy(data['context'].reshape(-1,35))
        collected={key:[] for key in ['prediction','normalized_loss','true_score','pred_score','true_style','pred_style','true_slope','pred_slope']}
        with torch.no_grad():
            for start in range(0,len(states),args.batch_size):
                index=slice(start,start+args.batch_size)
                state,truth=states[index],target[index]
                predicted=model.step(state,actions[index],contexts[index])
                loss=model.loss(predicted,truth)
                f0,f1,fp=adapter.amp_features(state),adapter.amp_features(truth),adapter.amp_features(predicted)
                def scoring(next_features):
                    return network(torch.cat((norm(f0),norm(next_features)),-1)).squeeze(-1)
                dt,dp=scoring(f1),scoring(fp)
                values=[predicted,loss,dt,dp,mapper(dt),mapper(dp),mapper.score_slope(dt),mapper.score_slope(dp)]
                for key,value in zip(collected,values):collected[key].append(value.numpy())
        output={k:np.concatenate(v,axis=0) for k,v in collected.items()}
        state=states.numpy();truth=target.numpy();pred=output['prediction']
        def calculations(next_state):
            displacement=next_state[:,13:42]-state[:,13:42]
            average_velocity=.5*(next_state[:,42:71]+state[:,42:71])
            residual=displacement-.02*average_velocity
            qacc=(next_state[:,42:71]-state[:,42:71])/.02
            return residual,qacc
        true_residual,true_fd=calculations(truth); pred_residual,pred_fd=calculations(pred)
        masks={'all_valid':data['valid'] & ~data['boundary'],
               'recovery_active':data['valid'] & ~data['boundary'] & data['recovery_phase'],
               'walk_after_recovery':data['valid'] & ~data['boundary'] & ~data['recovery_phase'],
               'paired_fixed_0p1_to_2_seconds':common & window}
        sections={}
        for phase,mask in masks.items():
            choose=mask.reshape(-1)
            if not choose.any():continue
            real_acc=data['qacc_cost'].reshape(-1)[choose]
            true_fd_cost=np.square(true_fd[choose].astype(np.float64)).sum(-1)
            pred_fd_cost=np.square(pred_fd[choose].astype(np.float64)).sum(-1)
            sections[phase]={
              'transitions':int(choose.sum()),'physical_last_substep_qacc_cost':stats(real_acc),
              'true_velocity_fd_qacc_cost':stats(true_fd_cost),'model_velocity_fd_qacc_cost':stats(pred_fd_cost),
              'mean_model_fd_over_true_fd':float(pred_fd_cost.mean()/true_fd_cost.mean()),
              'mean_true_fd_over_actual_last_substep':float(true_fd_cost.mean()/real_acc.mean()),
              'mean_model_fd_over_actual_last_substep':float(pred_fd_cost.mean()/real_acc.mean()),
              'true_q_integral_residual_rad':stats(true_residual[choose]),
              'pred_q_integral_residual_rad':stats(pred_residual[choose]),
              'joint_position_prediction_error_rad':stats((pred-truth)[choose,13:42]),
              'joint_velocity_prediction_error_rad_s':stats((pred-truth)[choose,42:71]),
              'normalized_model_loss':stats(output['normalized_loss'][choose]),
              'true_style':stats(output['true_style'][choose]),'predicted_style':stats(output['pred_style'][choose]),
              'style_h1_prediction_minus_true':stats((output['pred_style']-output['true_style'])[choose]),
              'true_score':stats(output['true_score'][choose]),'predicted_score':stats(output['pred_score'][choose]),
              'true_mapping_zero_slope_fraction':float((output['true_slope'][choose]==0).mean()),
              'pred_mapping_zero_slope_fraction':float((output['pred_slope'][choose]==0).mean()),
              'true_mapping_slope':stats(output['true_slope'][choose]),
              'pred_mapping_slope':stats(output['pred_slope'][choose]),
            }
        # Algebraic control, not a feasible physical/model action perturbation:
        # only the q coordinates of the next-state tensor are perturbed.
        perturbed=pred.copy();perturbed[:,13:42]+=.05
        _,perturbed_fd=calculations(perturbed)
        result['policies'][name]={'data':str(args.recording/f'transitions_{name}.npz'),'sha256':sha(args.recording/f'transitions_{name}.npz'),
            'phase_results':sections,'structural_partial_qpos_control':{
                'next_joint_q_perturbation_rad':.05,'dq_held_fixed':True,
                'max_qacc_cost_change':float(np.max(np.abs(np.square(perturbed_fd).sum(-1)-np.square(pred_fd).sum(-1)))),
                'interpretation':'Direct qacc reward has zero partial derivative in next_q at fixed next_dq; shared action->q,dq model derivatives mean this does not prove actor can realize the perturbation.'}}
        print(f'completed {name}',flush=True)
    result['wall_seconds']=time.perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
