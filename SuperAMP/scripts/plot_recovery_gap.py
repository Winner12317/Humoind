"""Plot only saved diagnostic JSON; no simulator or training code is run."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/validation'
model=json.loads((OUT/'recovery_gap_model_40800_full.json').read_text())
style=json.loads((OUT/'recovery_gap_style_20261116.json').read_text())
reward=json.loads((OUT/'recovery_gap_reward_20261116.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors={'PPO60000':'#3468ad','WMAC40800':'#d27827'}
h=np.array(model['horizons'])
for scene,group,label,color in [('walk','all','Walking','#3468ad'),
                                ('recovery','recovery_phase','Recovery','#d27827')]:
    rows=[r['groups'][group]['metrics'] for r in model['recordings'] if r['mode']==scene]
    vals=[np.mean([r[str(x)]['joint_position_rmse_rad'] for r in rows]) for x in h]
    axs[0,0].plot(h,vals,'o-',label=label,color=color)
axs[0,0].set(title='A. Frozen world model: recovery error is larger',
             xlabel='Open-loop prediction steps (20 ms each)',ylabel='Joint position RMSE (rad)')
axs[0,0].legend()
rec=style['recordings']['ppo60000_vs_wmac40800_recovery_seed2026']['trials']
walk=style['recordings']['ppo60000_vs_wmac40800_walk_seed2026']['trials']
def zero(trials,policy,phase):
    return 100*np.mean([trials[t]['phases'][phase]['judge']['wm40800']['zero_gradient_fraction']
                        for t in (policy,policy+'_repeat')])
zs=[zero(walk,'b','all'),zero(rec,'a','replay_recovery_phase'),zero(rec,'b','replay_recovery_phase')]
bars=axs[0,1].bar(['WM walking','PPO recovery','WM recovery'],zs,
                 color=['#3468ad','#7f9ebd','#d27827'])
axs[0,1].bar_label(bars,fmt='%.1f%%')
axs[0,1].set(title='B. Same frozen D40800: zero AMP reward / direct gradient',
             ylabel='Outside AMP reward support (%)',ylim=(0,40))
entry=next(r for r in reward['runs'] if '40800_recovery' in r['directory'])
g=entry['groups']['fixed_0p1_to_2_seconds']
rr=[g[k]['repeat_average_then_equal_initial_state_mean'] for k in ('a','b')]
x=np.arange(2)
height=np.array([r['height_positive_reward'] for r in rr])
loc=np.array([r['locomotion_positive_reward_group'] for r in rr])
pen=np.array([r['known_penalties_reward'] for r in rr])
axs[1,0].bar(x,height,label='Recovery height',color='#9fb58f')
axs[1,0].bar(x,loc,bottom=height,label='Locomotion / stabilization',color='#87adcc')
axs[1,0].bar(x,pen,label='Known costs',color='#dca68b')
axs[1,0].scatter(x,[r['real_task_reward'] for r in rr],c='black',label='Net task reward',zorder=3)
axs[1,0].set(xticks=x,xticklabels=['PPO60000','WMAC40800'],
             title='C. Actual task reward favors the faster recovery',ylabel='Task reward per 20 ms (0.1-2 s)')
axs[1,0].legend(fontsize=8)
rows=[r['groups']['recovery_phase']['metrics']['1'] for r in model['recordings'] if r['mode']=='recovery']
names=['foot_slip','self_collisions'];true=[np.mean([r[n]['physical_mean'] for r in rows]) for n in names]
pred=[np.mean([r[n]['predicted_mean'] for r in rows]) for n in names]
axs[1,1].bar(np.arange(2)-.18,true,.36,label='Recorded physics',color='#3468ad')
axs[1,1].bar(np.arange(2)+.18,pred,.36,label='Existing model',color='#d27827')
axs[1,1].set(xticks=np.arange(2),xticklabels=['Foot slip','Self collisions'],
             title='D. Existing contact-cost predictions are low in recovery',ylabel='Raw one-step cost (different definitions)')
axs[1,1].legend()
fig.suptitle('PPO60000 vs WMAC40800: paired physical diagnosis, no training changes',fontsize=13)
for ext in ('png','svg'):fig.savefig(OUT/f'recovery_gap_40800.{ext}',dpi=180)
print(OUT/'recovery_gap_40800.png')
