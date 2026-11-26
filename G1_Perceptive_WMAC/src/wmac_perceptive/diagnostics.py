"""Detached device reductions, with one host read per diagnostic dictionary."""
from __future__ import annotations

import torch

from src.wmac.routing import is_moe, load_fraction, router_losses, routing_metric_keys


def read_metrics(values):
    if not values:return {}
    reference=next(value for value in values.values() if isinstance(value,torch.Tensor))
    packed=torch.stack([torch.as_tensor(value,device=reference.device,dtype=torch.float64)
                        for value in values.values()])
    return dict(zip(values,packed.detach().cpu().tolist()))


def masked_mean(value,mask,*,dim=None,empty=float("inf")):
    mask=mask.bool()
    count=mask.sum(dim=dim)
    total=torch.where(mask,value,torch.zeros_like(value)).sum(dim=dim)
    return torch.where(count>0,total/count.clamp_min(1),total.new_full((),empty))


@torch.no_grad()
def routing_tensors(actor,observations,recovery_mask):
    result={key:observations.new_zeros(()) for key in routing_metric_keys(actor)}
    if not is_moe(actor):return result,None
    balance,z,route=router_losses(actor,observations)
    p=route.probabilities.reshape(-1,actor.num_experts)
    selected=route.indices.reshape(-1,actor.top_k)
    fraction=load_fraction(selected,actor.num_experts,p.dtype)
    result.update(moe_load_balance_loss=balance,moe_router_z_loss=z,
        moe_router_entropy=-(p*p.clamp_min(1e-12).log()).sum(-1).mean(),
        moe_load_entropy=-(fraction*fraction.clamp_min(1e-12).log()).sum(),
        moe_unused_experts_in_batch=(fraction==0).sum())
    if actor.top_k<actor.num_experts:
        ranked=p.topk(actor.top_k+1,dim=-1).values
        result['moe_topk_margin']=(ranked[:,-2]-ranked[:,-1]).mean()
    recovery_mask=recovery_mask.reshape(-1).bool()
    result['moe_recovery_sample_fraction']=recovery_mask.float().mean()
    assignments=torch.nn.functional.one_hot(selected,actor.num_experts).to(p.dtype).mean(1)
    for group,mask in (('',torch.ones_like(recovery_mask)),('walk_',~recovery_mask),
                       ('recovery_',recovery_mask)):
        usage=(assignments*mask[:,None]).sum(0)/mask.sum().clamp_min(1)
        for index in range(actor.num_experts):result[f'moe_{group}load_{index}']=usage[index]
    return result,route


def grad_norm(module):
    terms=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return torch.stack(terms).sum().sqrt() if terms else next(module.parameters()).new_zeros(())


@torch.no_grad()
def validation_tensors(runner,values,mask,recovery,physical_metrics):
    """Keep all keys, masks and prefix rules; reduce columns on the device."""
    result={}
    for name in physical_metrics:
        if name in values:result[f'validation_{name}']=masked_mean(values[name],mask)
    if runner.uses_latent:
        for name in ('latent_error','latent_persistence_error'):
            result[f'validation_{name}']=masked_mean(values[name],mask)
    for region in ('full','actor','outside'):
        valid=mask.bool()&(values[f'{region}_valid_pixels']>0)
        missing=float('nan') if runner.uses_latent and region!='actor' else 0.
        result[f'validation_{region}_depth_rmse']=masked_mean(values[f'{region}_rmse'],valid,empty=missing)
        result[f'validation_{region}_depth_samples']=valid.sum()
    selections={'walk':~recovery,'recovery':recovery}
    for phase in runner._motion_phases():
        valid=selections[phase][:,None]&mask.bool()
        depth_valid=valid&(values['actor_valid_pixels']>0)
        columns={
            'model_error':masked_mean(values['model_error'],valid,dim=0),
            'contact_error':masked_mean(values['contact_error'],valid,dim=0),
            'actor_depth_rmse':masked_mean(values['actor_rmse'],depth_valid,dim=0,empty=0.),
            'actor_mask_error':masked_mean(values['actor_mask_error'],valid,dim=0)}
        if runner.uses_latent:
            for name in ('latent_error','latent_persistence_error'):
                columns[name]=masked_mean(values[name],valid,dim=0)
        prefixes={}
        for name,column in columns.items():
            finite=torch.isfinite(column).to(torch.int32).cumprod(0).bool()
            prefixes[name]=torch.where(finite,column.cummax(0).values,column.new_full((),float('inf')))
        counts=valid.sum(0)
        depth_counts=depth_valid.sum(0)
        prefix_count=counts.cumsum(0)
        finite_count=(valid&(values['finite']==1)).sum(0).cumsum(0)
        fraction=torch.where(prefix_count>0,finite_count.double()/prefix_count.clamp_min(1),0.)
        regions={}
        for region in ('full','actor','outside'):
            region_valid=valid&(values[f'{region}_valid_pixels']>0)
            missing=float('nan') if runner.uses_latent and region!='actor' else 0.
            regions[region]=(masked_mean(values[f'{region}_rmse'],region_valid,dim=0,empty=missing),
                             region_valid.sum(0))
        for index in range(values['model_error'].shape[1]):
            step=index+1
            for region,(errors,samples) in regions.items():
                result[f'validation_{phase}_step{step}_{region}_depth_rmse']=errors[index]
                result[f'validation_{phase}_step{step}_{region}_depth_samples']=samples[index]
            prefix=f'validation_{phase}_h{step}_'
            result[prefix+'samples']=counts[index]
            result[prefix+'depth_samples']=depth_counts[index]
            result[prefix+'finite_fraction']=fraction[index]
            for name,column in prefixes.items():result[prefix+name]=column[index]
    return result
