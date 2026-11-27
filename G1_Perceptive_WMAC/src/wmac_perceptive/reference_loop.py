"""SuperMimic collection clock with the independent Hiking discriminator rollout."""

from time import perf_counter

from .compact_gpu_replay import NoCompleteReferenceWindow


def learn_reference(runner):
    """Count vector steps as in SuperMimic's runner and Ratio scheduler."""
    cfg = runner.cfg
    start_step = runner.sim_steps // runner.env.num_envs
    ratio = 1. / cfg.collect_steps if cfg.collect_steps else cfg.utd
    collection = {}
    collect_seconds = disc_seconds = 0.
    collected = 0
    disc_counts = {}
    last_disc = {}
    initial_iteration = runner.iteration
    last_report_iteration = initial_iteration
    last_report_wall = runner.wall_time
    first_missing_window_step = None
    last_window_error = ""
    for step in range(start_step, cfg.iterations):
        iteration_before_step = runner.iteration
        started = perf_counter()
        for key, value in runner.collect(1).items():
            collection[key] = collection.get(key, 0.) + value
        collected += 1
        after_collect = perf_counter()
        collect_seconds += after_collect - started
        disc = runner.update_discriminator()
        for key in ("discriminator_updates_applied", "amp_rollout_blocks", "amp_policy_windows",
                    "amp_policy_exposures", "amp_invalid_windows_dropped"):
            disc_counts[key] = disc_counts.get(key, 0) + disc[key]
        if disc["discriminator_updates_applied"] or not last_disc:
            last_disc = disc.copy()
        disc_seconds += perf_counter() - after_collect

        # The first event fires immediately and is skipped while Replay fills.
        # Advance the ratio clock even when the learner cannot update yet.
        repeats = 1 if step == 0 else int((step - runner._collect_accumulator) * ratio)
        if step:
            runner._collect_accumulator += repeats / ratio
        if runner.replay.size >= cfg.wm_window + 1:
            for _ in range(repeats):
                begun = perf_counter()
                try:
                    row = runner.fit_model()
                except NoCompleteReferenceWindow as exc:
                    # Replay is constant across all updates at this vector step.
                    # The sampling exception is raised before any model update.
                    last_window_error = str(exc)
                    if first_missing_window_step is None:
                        first_missing_window_step = step
                        print(f"[WMAC] vector_step={step + 1}: {exc}. Continuing collection.", flush=True)
                    elif step - first_missing_window_step >= runner.replay.capacity_steps:
                        raise RuntimeError(
                            f"World-model training could not resume by vector step {step + 1}. "
                            f"First unavailable update at step {first_missing_window_step + 1}; "
                            f"at least one full replay buffer has since been collected. {exc}"
                        ) from exc
                    break
                first_missing_window_step = None
                after_model = perf_counter()
                imagined_before = runner.imagination_steps
                policy = runner.update_actor_critic(update_actor=True)
                if cfg.actor_updates > 1:
                    totals = policy.copy()
                    for _ in range(1, cfg.actor_updates):
                        for key, value in runner.update_actor_critic(update_actor=True).items():
                            totals[key] += value
                    policy = {key: value / cfg.actor_updates for key, value in totals.items()}
                row.update(policy)
                if cfg.validation_env_fraction and runner.iteration % cfg.validation_interval == 0:
                    runner.validate_model()
                finished = perf_counter()
                runner.iteration += 1
                row.update({key: value / max(collected, 1) for key, value in collection.items()})
                row.update(disc)
                for key in ("discriminator_loss", "discriminator_grad_penalty", "discriminator_grad_norm"):
                    row[key] = last_disc.get(key, 0.)
                row.update(disc_counts)
                row.update(runner.validation if cfg.validation_env_fraction else {"validation_model_error": float("nan")})
                row.update(vector_step=step + 1, collect_steps=collected, collect_seconds=collect_seconds,
                           disc_seconds=disc_seconds, model_seconds=after_model-begun,
                           actor_critic_seconds=finished-after_model, actor_active=1.,
                           learning_rate=cfg.actor_lr, gate_reason="supermimic_schedule",
                           physical_steps_per_second=collected*runner.env.num_envs/max(
                               collect_seconds + disc_seconds + finished-begun, 1e-8),
                           imagined_steps_per_second=(runner.imagination_steps-imagined_before)/max(
                               finished-after_model, 1e-8))
                runner._log(row)
                if runner.iteration == initial_iteration + 1 or runner.iteration % cfg.log_interval == 0:
                    wall = runner.wall_time
                    seconds_per_iter = (wall - last_report_wall) / (runner.iteration - last_report_iteration)
                    runner.writer.flush()
                    print(
                        f"[WMAC] iter={runner.iteration} sim={runner.sim_steps} imag={runner.imagination_steps} "
                        f"wall={wall:.1f}s wm={row['wm_loss']:.3g} val={row['validation_model_error']:.3g} "
                        f"actor={row['actor_loss']:.3g} active={int(row['actor_active'])} "
                        f"sec_per_iter={seconds_per_iter:.2f} vector_step={step+1}",
                        flush=True,
                    )
                    last_report_iteration = runner.iteration
                    last_report_wall = wall

                collection = dict.fromkeys(collection, 0.)
                disc_counts = dict.fromkeys(disc_counts, 0)
                collected = 0
                collect_seconds = disc_seconds = 0.
        if first_missing_window_step is not None and step + 1 == cfg.iterations:
            raise RuntimeError(
                f"World-model training could not resume by vector step {step + 1}. "
                f"First unavailable update at step {first_missing_window_step + 1}; "
                f"last attempted update: {last_window_error}"
            )
        # Finish all updates for this control step before saving so the
        # checkpoint's simulation and update clocks can resume together.
        crossed_save_interval = (
            runner.iteration // cfg.save_interval > iteration_before_step // cfg.save_interval
        )
        if step == 0 or crossed_save_interval or step + 1 == cfg.iterations:
            runner.save(runner.log_dir / f"model_{runner.iteration}.pt")
