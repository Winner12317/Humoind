"""InstinctMJ visual-latent training adapted to this task's GenCoord and AMP.

Real processed image stacks remain in replay. The online policy encoder is
trained only by reconstruction; dynamics starts/targets are stopped gradients.
Imagined trajectories predict a joint stack latent without rendering/decoding.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from src.wmac.device_placement import boundary
from src.wmac.runner import _device_batch
from src.wmac.networks import _quat_to_rotvec
from src.wmac.state import quat_inv, quat_mul
from src.tasks.amp_perceptive.sensors import read_depth_frame
from .perception import advance_depth_ring,select_depth_frames
from .latent_perception import processed_depth_stack
from .instinct_latent import normalized_latent_errors


@torch.no_grad()
def collect_latent(runner, count):
    totals = dict(real_task_reward=0., terminated_fraction=0., timeout_fraction=0.,
                  recovery_fraction=0., boundary_fraction=0., invalid_transition_fraction=0.,
                  sensor_valid_fraction=0., standing_command_fraction=0.)
    fast_collection=getattr(runner.cfg,"hiking_training_version",1)>=2
    # The simulator may live on another device than the learner; then these two
    # move what crosses per control step. On one device they return their argument.
    from_env,to_env=boundary(runner)
    if fast_collection:
        # Each step still contributes its own mean. Keep the small diagnostics
        # on the device until this collection finishes, instead of reading one
        # CUDA scalar for every metric after every physical step.
        device_totals=torch.zeros(len(totals),dtype=torch.float64,
            device=getattr(runner,"device",None) or runner.env.obs_buf["actor"].device)
    reuse_processed=(fast_collection and getattr(runner,"_processed_history",False)
        and runner.spec.depth_protocol=="hiking_v1"
        and runner.cfg.depth_noise_std_m==0 and runner.cfg.depth_dropout_probability==0)
    reduced_history=(reuse_processed and runner.depth_history.spec.delay==1
        and all(callable(getattr(runner.depth_history,name,None)) for name in
                ("sampled_frames","sampled_successor_frames","append_with_reset_ids")))
    for _ in range(count):
        proprio = torch.nan_to_num(from_env(runner.env.obs_buf["actor"]))
        # append() replaces its buffer rather than mutating this snapshot.
        # The optimized history already contains processed single frames.
        processed_history=getattr(runner,"_processed_history",False)
        ring = runner.depth_history.ring if processed_history else runner.depth_history.snapshot()
        stack = (runner.depth_history.sampled_frames().flatten(-4,-3) if reduced_history else
            select_depth_frames(ring,runner.depth_history.spec).flatten(-4,-3)
            if processed_history else processed_depth_stack(ring,runner.spec))
        if processed_history and not runner.spec.include_valid:
            stack=stack[...,0::2,:,:]
        latent = runner.actor.encode_processed(stack)
        action = runner.actor.forward_latent(proprio, latent)
        action = action + runner.cfg.exploration_noise * torch.randn_like(action)
        if runner.cfg.action_limit is not None:
            action = action.clamp(-runner.cfg.action_limit, runner.cfg.action_limit)
        if not bool(torch.isfinite(action).all()):
            raise RuntimeError("Non-finite latent policy action")
        runner.env.step(to_env(action))
        batch = from_env(runner.env.transition)
        if not fast_collection and "command_standing" in batch:
            totals["standing_command_fraction"] += float(batch["command_standing"].float().mean())
        # Capture the end-of-transition stack before episode reset. Reuse its
        # one corruption draw for the next real policy on nonterminal steps.
        end_frame = runner._policy_depth_frame(batch["next_depth_frame"].float())
        if reduced_history:
            next_stack=runner.depth_history.sampled_successor_frames(end_frame).flatten(-4,-3)
        else:
            end_ring = advance_depth_ring(ring, end_frame, runner.depth_history.spec)
            next_stack=(select_depth_frames(end_ring,runner.depth_history.spec).flatten(-4,-3)
                        if processed_history else processed_depth_stack(end_ring,runner.spec))
        if processed_history and not runner.spec.include_valid:
            next_stack=next_stack[...,0::2,:,:]
        stored = dict(batch)
        stored.pop("next_depth_frame")
        direct_uint8=(not runner.spec.include_valid
                      and getattr(runner.cfg,"replay_storage","legacy")=="frame_uint8")
        stored.update(depth_stack=stack if direct_uint8 else stack.to(torch.float16),
                      next_depth_stack=next_stack if direct_uint8 else next_stack.to(torch.float16))
        accepted = runner.replay.add(stored)
        train = accepted & runner._train_env_mask
        runner.world_model.update_normalizers_latent(batch["state"][train], batch["action"][train],
            batch["context"][train], batch["next_state"][train], latent[train])
        if not runner.uses_reference_schedule:
            runner.critic.norm.update(runner.adapter.critic_features(batch["state"][train],
                batch["previous_action"][train], batch["command"][train], batch["context"][train]))
        runner.current_depth = from_env(read_depth_frame(runner.env)).detach()
        done = batch["terminated"] | batch["timeout"]
        if reuse_processed:
            # Physical DR has already drawn exactly once for each captured
            # frame. Only reset rows have a distinct post-reset camera image;
            # the other rows can use the preprocessed physical successor.
            reset_ids=done.nonzero(as_tuple=False).flatten()
            policy_frame=end_frame
            if reset_ids.numel():
                policy_frame=end_frame.clone()
                policy_frame[reset_ids]=runner._policy_depth_frame(runner.current_depth[reset_ids])
        else:
            reset_frame = runner._policy_depth_frame(runner.current_depth)
            policy_frame = torch.where(done[:, None, None, None], reset_frame, end_frame)
        if reduced_history:
            runner.depth_history.append_with_reset_ids(policy_frame,reset_ids)
        else:
            runner.depth_history.append(policy_frame, reset_mask=done)
        runner.sim_steps += runner.env.num_envs
        if fast_collection:
            accepted_count=accepted.sum()
            reward=torch.where(accepted,batch["reward"],torch.zeros_like(batch["reward"])).sum()
            reward=reward/accepted_count.clamp_min(1)
            standing=(batch["command_standing"].float().mean() if "command_standing" in batch
                      else reward.new_zeros(()))
            device_totals+=torch.stack((reward,batch["terminated"].float().mean(),
                batch["timeout"].float().mean(),reward.new_zeros(()),
                batch["boundary"].float().mean(),(~accepted).float().mean(),
                runner.current_depth[:,1].mean(),standing)).to(torch.float64)
            continue
        totals["real_task_reward"] += float(batch["reward"][accepted].mean()) if accepted.any() else 0.
        for name, key in (("terminated_fraction", "terminated"), ("timeout_fraction", "timeout"),
                          ("boundary_fraction", "boundary")):
            totals[name] += float(batch[key].float().mean())
        totals["invalid_transition_fraction"] += float((~accepted).float().mean())
        totals["sensor_valid_fraction"] += float(runner.current_depth[:, 1].mean())
    if fast_collection:
        totals=dict(zip(totals,device_totals.cpu().tolist()))
    return {key: value/max(count, 1) for key, value in totals.items()}


def decoded_stack_metrics(prediction, target, max_depth, *, include_valid=True):
    """Meter diagnostics for the same processed stack represented by the z.

Full raw/outside images are intentionally unavailable: do not report a perfect
zero for regions absent from this observation protocol.
"""
    if not include_valid:
        # Depth-only reconstruction includes the reference's filled no-hit
        # pixels. A validity prediction is unavailable, not a perfect zero.
        error=(prediction-target).square().mean((-3,-2,-1)).sqrt()*max_depth
        missing=torch.full_like(error,float("nan"))
        count=torch.full_like(error,target.shape[-3]*target.shape[-2]*target.shape[-1])
        return dict(actor_rmse=error, actor_mask_error=missing, actor_valid_pixels=count,
                    full_rmse=missing, outside_rmse=missing, full_mask_error=missing,
                    full_valid_pixels=torch.zeros_like(count),outside_valid_pixels=torch.zeros_like(count))
    depth, valid = prediction[..., 0::2, :, :], prediction[..., 1::2, :, :]
    truth, hit = target[..., 0::2, :, :], target[..., 1::2, :, :].clamp(0, 1)
    axes = (-3, -2, -1)
    pixels = hit.sum(axes)
    error = (((depth-truth).square()*hit).sum(axes)/pixels.clamp_min(1)).sqrt()*max_depth
    missing = torch.full_like(error, float("nan"))
    return dict(actor_rmse=error, actor_mask_error=(valid-hit).square().mean(axes),
                actor_valid_pixels=pixels, full_rmse=missing, outside_rmse=missing,
                full_mask_error=missing, full_valid_pixels=torch.zeros_like(pixels),
                outside_valid_pixels=torch.zeros_like(pixels))


def rollout_latent_model_metrics(runner, batch, *, update_statistics=False):
    from .runner import PHYSICAL_METRICS
    state = batch["state"][:, 0]
    context=batch["context"][:,0]
    with torch.no_grad():
        latent = runner.actor.encode_processed(batch["depth_stack"][:, 0].float()).detach()
        target = runner.actor.encode_processed(batch["next_depth_stack"].flatten(0, 1).float())
        target = target.reshape(*batch["mask"].shape, -1).detach()
        if update_statistics:
            previous = torch.cat((latent[:, None], target[:, :-1]), 1)
            runner.world_model.update_z_stats(target, target-previous, batch["mask"],
                                               rate=runner.cfg.latent_stat_rate)
    initial_latent = latent
    decode_diagnostics = not torch.is_grad_enabled()
    records, predicted_latents = [], []
    for step in range(batch["mask"].shape[1]):
        active = batch["mask"][:, step].bool()
        state = torch.where(active[:, None], state, batch["state"][:, step])
        if runner.uses_full_hiking_task:
            context=torch.where(active[:,None],context,batch["context"][:,step])
        else:
            context=batch["context"][:,step]
        state, contact, latent = runner.world_model.step_with_latent(
            state, batch["action"][:, step], context, latent)
        if runner.uses_full_hiking_task:
            from .hiking_context import advance_hiking_context
            context=advance_hiking_context(context,batch["action"][:,step],contact,
                camera_capture_steps=5 if getattr(runner.cfg,"sensor_protocol",None)=="g1_rpl_depth_v1" else 0)
        predicted_latents.append(latent)
        with torch.no_grad():
            truth = batch["next_state"][:, step]
            errors = [(state[..., section]-truth[..., section]).square().mean(-1).sqrt()
                      for section in (slice(0, 3), slice(7, 10), slice(10, 13), slice(13, 42), slice(42, 71))]
            orientation = _quat_to_rotvec(quat_mul(quat_inv(truth[..., 3:7]), state[..., 3:7])).norm(dim=-1)
            physical = dict(zip(PHYSICAL_METRICS, (errors[0], orientation, *errors[1:])))
            finite = (torch.isfinite(state).all(-1) & torch.isfinite(contact).all(-1)
                      & torch.isfinite(latent).all(-1))
            if decode_diagnostics:
                decoded = runner.latent_decoder(latent)
                visual = decoded_stack_metrics(decoded, batch["next_depth_stack"][:, step].float(),
                    1.9 if runner.spec.depth_protocol=="g1_rpl_v1" else runner.spec.max_depth,
                    include_valid=runner.spec.include_valid)
            else:
                missing = latent.new_full(latent.shape[:-1], float("nan"))
                visual = {name: missing for name in (
                    "full_rmse", "actor_rmse", "outside_rmse", "full_mask_error", "actor_mask_error")}
                visual.update({name: torch.zeros_like(missing) for name in (
                    "actor_valid_pixels", "full_valid_pixels", "outside_valid_pixels")})
        records.append(dict(model_error=runner.world_model.loss(state, batch["next_state"][:, step]),
            contact_error=F.smooth_l1_loss(torch.log1p(contact),
                torch.log1p(batch["contact_cost"][:, step].clamp_min(0)), reduction="none").mean(-1),
            finite=finite.float(), **visual, **physical))
    result = {key: torch.stack([row[key] for row in records], 1) for key in records[0]}
    predicted = torch.stack(predicted_latents, 1)
    result["latent_error"] = normalized_latent_errors(predicted, target)
    result["latent_persistence_error"] = normalized_latent_errors(initial_latent[:, None].expand_as(target), target)
    # Supervision and reconstruction always use the current online E, not
    # stale codes cached before its most recent optimizer step.
    result["target_latents"] = target
    return result


def fit_latent_model(runner):
    if runner.uses_fast_execution:
        return fit_latent_model_fast(runner)
    from .runner import PHYSICAL_METRICS
    names = ("wm_loss", "wm_contact_loss", "wm_last_error", "wm_grad_norm", "wm_walk_error",
             "wm_updates_applied_fraction", "wm_latent_loss", "wm_latent_recon_l1",
             "wm_latent_persistence_error", "wm_latent_target_std",
             *(f"wm_h{h}_model_error" for h in range(1, runner.cfg.wm_window+1)),
             *(f"wm_h{h}_latent_error" for h in range(1, runner.cfg.wm_window+1)),
             *(f"wm_{name}" for name in PHYSICAL_METRICS))
    records = []
    for _ in range(runner.cfg.wm_updates):
        batch = _device_batch(runner.replay.sample_windows(runner.cfg.wm_batch_size, runner.cfg.wm_window,
            0., fields=runner.WM_FIELDS), runner.device)
        values = rollout_latent_model_metrics(runner, batch, update_statistics=True)
        mask = batch["mask"]
        mean = lambda value: runner._masked_mean(value, mask)
        dyn, contact = (mean(values[key]) for key in ("model_error", "contact_error"))
        # Match the reference k-step latent objective: sum valid steps / B.
        # The physical objective retains this project's GenCoord mean scaling.
        latent = (values["latent_error"]*mask).sum()/len(mask)
        stack = batch["next_depth_stack"][mask.bool()].float()[:runner.cfg.latent_recon_images]
        reconstruction = (runner.latent_decoder(runner.actor.encode_processed(stack))-stack).abs().mean()
        norm, ok = runner._optimizer_step(dyn+contact+runner.cfg.latent_loss_coef*latent
            +runner.cfg.latent_recon_coef*reconstruction, runner.opt_model, runner._model_training_modules)
        with torch.no_grad():
            target = values["target_latents"][mask.bool()]
            target_std = target.std(0, unbiased=False).mean() if target.numel() else dyn.new_tensor(float("nan"))
            prefixes = [runner._masked_mean(values["model_error"][:, :h], mask[:, :h])
                        for h in range(1, runner.cfg.wm_window+1)]
            latent_prefixes = [runner._masked_mean(values["latent_error"][:, :h], mask[:, :h])
                               for h in range(1, runner.cfg.wm_window+1)]
            records.append(torch.stack((dyn.detach(), contact.detach(),
                runner._masked_mean(values["model_error"][:, -1], mask[:, -1]), norm, dyn.detach(),
                dyn.new_tensor(float(ok)), mean(values["latent_error"]), reconstruction.detach(),
                mean(values["latent_persistence_error"]), target_std, *prefixes, *latent_prefixes,
                *(mean(values[name]) for name in PHYSICAL_METRICS))))
    result = dict(zip(names, torch.stack(records).mean(0).detach().cpu().tolist()))
    runner.last_model_loss = result["wm_loss"]
    result["wm_gate_error"] = result["wm_loss"] if math.isfinite(result["wm_loss"]) else float("inf")
    return result


def fit_latent_model_fast(runner):
    """Train fixed tensor kernels; read the ten-update summary once."""
    execution=runner._latent_wm_execution
    records=[]
    sampler=(runner.replay.sample_reference_model_windows if getattr(runner.cfg, "hiking_training_version", 1) >= 3
             else getattr(runner.replay,"sample_model_windows",runner.replay.sample_windows))
    def sample():
        return _device_batch(sampler(runner.cfg.wm_batch_size,runner.cfg.wm_window,
            0.,fields=runner.WM_FIELDS),runner.device)
    if hasattr(execution,"step_macro"):
        matrix=execution.step_macro([sample() for _ in range(runner.cfg.wm_updates)])
    else:
        for _ in range(runner.cfg.wm_updates):
            metrics,norm,accepted=execution.step(sample())
            records.append(torch.cat((metrics,norm.reshape(1),accepted.float().reshape(1))))
        matrix=torch.stack(records)
    names=(*execution.metric_names,"wm_grad_norm","wm_updates_applied_fraction")
    result=dict(zip(names,matrix.mean(0).detach().cpu().tolist()))
    runner.last_model_loss=result["wm_loss"]
    result["wm_gate_error"]=result["wm_loss"] if math.isfinite(result["wm_loss"]) else float("inf")
    return result
