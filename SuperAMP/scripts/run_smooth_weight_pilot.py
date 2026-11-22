"""Bounded, matched-budget coefficient experiment; never starts the main run.

Uses the unchanged full-size networks and update method with a smaller sampler
and Actor batch. Defaults are screening settings, not production validation.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
WEIGHTS={
    "baseline":(.01,.00025,.00025,.000025),
    "gentle":(.015,.0005,.0005,.000025),
    "moderate":(.0175,.000625,.000625,.00003125),
    "balanced_v2":(.02,.00075,.00075,.0000375),
    "velocity_only":(.01,.001,.00025,.000025),
}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--variant",choices=WEIGHTS,required=True)
    parser.add_argument("--updates",type=int,default=500)
    parser.add_argument("--num-envs",type=int,default=512)
    parser.add_argument("--actor-batch-size",type=int,default=1024)
    parser.add_argument("--actor-microbatch-size",type=int,default=0)
    parser.add_argument("--compile",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--save-interval",type=int)
    parser.add_argument("--log-root",type=Path,default=ROOT/"logs/weight_pilots")
    parser.add_argument("--memory-fraction",type=float,default=.30)
    args=parser.parse_args()
    if args.updates<1 or not 0<args.memory_fraction<=.5:parser.error("Positive updates and memory fraction in (0,.5] required")
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError("This bounded physical training pilot requires CUDA")
    free,total=torch.cuda.mem_get_info(0)
    if free<8*1024**3:raise RuntimeError("Less than 8 GiB GPU memory free; main training is left running")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device=0)
    checkpoint=args.checkpoint.expanduser().resolve()
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if payload.get("config",{}).get("smooth_reward_mode")!="host_humanup_v1":raise ValueError("Pilot source must already use current smooth formula")
    source_iteration=int(payload["iteration"])
    source_weights=tuple(payload["config"][name] for name in ("smooth_action_second_coef","smooth_joint_velocity_coef","smooth_torque_coef","smooth_power_coef"))
    if source_weights!=WEIGHTS["baseline"]:raise ValueError("All pilots must branch from the same original coefficient source")
    del payload
    from scripts.train_wmac import parse_config,main as train_main
    argv=["--resume",str(checkpoint),"--num-envs",str(args.num_envs),
          "--actor-batch-size",str(args.actor_batch_size),"--actor-microbatch-size",str(args.actor_microbatch_size),"--iterations",str(source_iteration+args.updates),
          "--compile" if args.compile else "--no-compile","--evaluation-interval","0","--save-interval",str(args.save_interval or args.updates),
          "--log-interval","20","--run-name",f"weight_pilot_{args.variant}_{args.updates}",
          "--log-root",str(args.log_root)]
    if args.variant!="baseline":
        argv += ["--allow-smooth-reward-retune"]
        for name,value in zip(("action-second","joint-velocity","torque","power"),WEIGHTS[args.variant]):
            argv += ["--smooth-"+name+"-coef",str(value)]
    cfg=parse_config(argv)
    print("[weight-pilot] "+json.dumps({"variant":args.variant,"source":str(checkpoint),
        "source_iteration":source_iteration,"additional_updates":args.updates,"num_envs":args.num_envs,
        "actor_batch_size":args.actor_batch_size,"actor_microbatch_size":cfg.actor_microbatch_size,
        "wm_batch_size":cfg.wm_batch_size,"network_shapes_unchanged":True,
        "wm_updates":cfg.wm_updates,"horizon":cfg.imagination_horizon,"utd":cfg.utd,
        "torch_memory_fraction":args.memory_fraction,"weights":WEIGHTS[args.variant],
        "scope":"Screening pilot; production keeps 4096 envs and Actor batch4096", "compile":cfg.compile},sort_keys=True),flush=True)
    started=time.perf_counter()
    train_main(cfg)
    print("[weight-pilot] "+json.dumps({"completed":True,"wall_seconds":time.perf_counter()-started,
        "peak_torch_allocated_bytes":torch.cuda.max_memory_allocated(0),
        "peak_torch_reserved_bytes":torch.cuda.max_memory_reserved(0)}),flush=True)
if __name__=="__main__":main()
