"""Router regularization on fixed real observations, separate from task returns.

The hard expert assignments are detached. A full-batch load fraction can be
shared across microbatches, preserving the full-batch Switch-style gradient.
These are Actor regularizers, never rewards or Critic targets. Sparse selection
boundaries are not differentiated by this loss or by the Actor task gradient.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


BASE_ROUTING_METRICS = (
    'actor_task_loss', 'actor_router_aux_loss', 'moe_load_balance_loss',
    'moe_router_z_loss', 'moe_router_entropy', 'moe_load_entropy',
    'moe_unused_experts_in_batch', 'moe_topk_margin', 'moe_recovery_sample_fraction',
    'moe_router_grad_norm', 'moe_expert_grad_norm_mean', 'moe_expert_grad_norm_max',
)


def is_moe(actor):
    return hasattr(actor, 'route') and hasattr(actor, 'experts')


def routing_metric_keys(actor):
    if actor is None:
        return []
    keys = list(BASE_ROUTING_METRICS)
    if is_moe(actor):
        keys += [f'moe_{group}load_{i}' for group in ('', 'walk_', 'recovery_')
                 for i in range(actor.num_experts)]
    return keys


def load_fraction(indices, num_experts, dtype=torch.float32):
    assignments = F.one_hot(indices.reshape(-1, indices.shape[-1]), num_experts).to(dtype)
    return assignments.mean((0, 1)).detach()


def router_losses(actor, observations, full_batch_load=None):
    """Return unweighted balancing/z losses and this fresh routing graph."""
    route = actor.route(observations)
    probabilities = route.probabilities.reshape(-1, actor.num_experts)
    fraction = (load_fraction(route.indices, actor.num_experts, probabilities.dtype)
                if full_batch_load is None else full_batch_load.detach())
    balance = actor.num_experts * (fraction * probabilities.mean(0)).sum()
    z = torch.logsumexp(route.logits / actor.temperature, dim=-1).square().mean()
    return balance, z, route


@torch.no_grad()
def routing_metrics(actor, observations, recovery_mask=None):
    """Detached batch utilization; phase labels are diagnostics, not router input."""
    result = {key: 0. for key in routing_metric_keys(actor)}
    if not is_moe(actor):
        return result
    balance, z, route = router_losses(actor, observations)
    p = route.probabilities.reshape(-1, actor.num_experts)
    selected = route.indices.reshape(-1, actor.top_k)
    fraction = load_fraction(selected, actor.num_experts, p.dtype)
    result.update(moe_load_balance_loss=float(balance), moe_router_z_loss=float(z),
                  moe_router_entropy=float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean()),
                  moe_load_entropy=float(-(fraction * fraction.clamp_min(1e-12).log()).sum()),
                  moe_unused_experts_in_batch=float((fraction == 0).sum()))
    if actor.top_k < actor.num_experts:
        ranked = p.topk(actor.top_k+1, dim=-1).values
        result['moe_topk_margin'] = float((ranked[:, -2]-ranked[:, -1]).mean())
    if recovery_mask is None:
        recovery_mask = torch.zeros(len(p), dtype=torch.bool, device=p.device)
    recovery_mask = recovery_mask.reshape(-1).bool()
    result['moe_recovery_sample_fraction'] = float(recovery_mask.float().mean())
    for group, mask in (('', torch.ones_like(recovery_mask)), ('walk_', ~recovery_mask),
                        ('recovery_', recovery_mask)):
        usage = load_fraction(selected[mask], actor.num_experts, p.dtype) if mask.any() else torch.zeros_like(fraction)
        for i in range(actor.num_experts):
            result[f'moe_{group}load_{i}'] = float(usage[i])
    return result


def parameter_grad_norm(module):
    squared = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(squared).sum().sqrt()) if squared else 0.
