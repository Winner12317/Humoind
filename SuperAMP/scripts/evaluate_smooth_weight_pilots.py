"""Run paired physics on completed bounded weight pilots; no learning occurs."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def pick_run(root:Path,variant:str,updates:int):
    candidates=[]
    for run in root.iterdir():
        path=run/"config.json"
        if not path.is_file():continue
        cfg=json.loads(path.read_text())
        if cfg.get("run_name")==f"weight_pilot_{variant}_{updates}":candidates.append(run)
    if len(candidates)!=1:raise ValueError(f"Expected exactly one {variant} pilot, found {candidates}")
    return candidates[0]

def ready_checkpoint(run:Path,iteration:int):
    completed=run/"runtime.json";checkpoint=run/f"model_{iteration}.pt"
    if not completed.is_file():return None
    record=json.loads(completed.read_text())
    if not record.get("completed"):raise RuntimeError(f"Pilot failed: {run}")
    if not checkpoint.is_file():raise RuntimeError(f"Completed pilot lacks checkpoint: {run}")
    return checkpoint

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root",type=Path,default=ROOT/"logs/weight_pilots")
    parser.add_argument("--output",type=Path,default=ROOT/"logs/weight_pilot500_eval")
    parser.add_argument("--updates",type=int,default=500)
    parser.add_argument("--analysis-prefix",type=Path)
    parser.add_argument("--control-variant",choices=("baseline","gentle"),default="baseline")
    parser.add_argument("--candidate-variant",choices=("balanced_v2","gentle","moderate","velocity_only"),default="balanced_v2")
    parser.add_argument("--run-updates",type=int)
    parser.add_argument("--workers",type=int,choices=(1,2,3),default=1)
    parser.add_argument("--source-iteration",type=int,default=32000)
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    if args.control_variant==args.candidate_variant:
        parser.error("Control and candidate variants must differ")
    a=pick_run(args.pilot_root,args.control_variant,args.run_updates or args.updates)
    b=pick_run(args.pilot_root,args.candidate_variant,args.run_updates or args.updates)
    end=args.source_iteration+args.updates
    while True:
        ca,cb=ready_checkpoint(a,end),ready_checkpoint(b,end)
        if ca is not None and cb is not None:break
        if not args.wait:raise RuntimeError("Both training pilots must complete before physical comparison")
        print("[weight-eval] waiting for both matched pilots to finish",flush=True)
        time.sleep(30)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={"a_control":str(ca),"b_candidate":str(cb),"control_variant":args.control_variant,
              "candidate_variant":args.candidate_variant,"new_training_updates":0,
              "scenarios":["recovery_seed2026","recovery_seed2027","walk_seed2026"],
              "num_envs":32,"steps":250,"repeats":2,"physical_task_reward":"upright_v2_without_training_smooth_terms"}
    if args.control_variant=="baseline":
        protocol["a_baseline"]=str(ca)  # Preserve the historical baseline-plan field.
    (args.output/"evaluation_plan.json").write_text(json.dumps(protocol,indent=2)+"\n")
    def evaluate_one(case):
        mode,seed=case
        name=f"{mode}_seed{seed}";log=args.output/f"{name}.log"
        command=[sys.executable,str(ROOT/"scripts/compare_centered_recovery_impact.py"),
                 "--checkpoint-a",str(ca),"--checkpoint-b",str(cb),"--mode",mode,
                 "--num-envs","32","--steps","250","--seed",str(seed),"--output",str(args.output/name)]
        print("[weight-eval] starting "+name,flush=True)
        with log.open("w") as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print("[weight-eval] completed "+name,flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(evaluate_one, (("recovery",2026),("recovery",2027),("walk",2026))))
    prefix=args.analysis_prefix or ROOT/f"docs/experiments/weight_pilot{args.updates}_physics"
    command=[sys.executable,str(ROOT/"scripts/analyze_smooth_weight_pilots.py"),
             "--root",str(args.output),"--output-prefix",str(prefix),
             "--expected-updates",str(args.updates),"--label-a",f"{args.control_variant}{args.updates}",
             "--label-b",f"{args.candidate_variant}_{args.updates}","--plot"]
    subprocess.run(command,cwd=ROOT,check=True)
    print("[weight-eval] paired physics and CPU analysis completed",flush=True)
if __name__=="__main__":main()
