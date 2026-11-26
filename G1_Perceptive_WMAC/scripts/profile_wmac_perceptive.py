"""Run the ordinary trainer with one explicitly requested profiler window.

Example: python scripts/profile_wmac_perceptive.py -- --device cuda:0 --num-envs 4096
Arguments after -- are forwarded unchanged to train_wmac_perceptive.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def arguments(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-wait-iterations",type=int,default=100)
    parser.add_argument("--profile-active-iterations",type=int,default=24)
    parser.add_argument("--profile-memory",action="store_true")
    if "--" not in argv:
        parser.parse_args(argv)  # Preserve the ordinary --help entrypoint.
        parser.error("Separate profiler options and unchanged training arguments with --")
    separator=argv.index("--")
    return parser.parse_args(argv[:separator]),argv[separator+1:]


def main(argv=None):
    options,training_arguments=arguments(argv)
    from scripts.train_wmac_perceptive import parse_config,main as train
    from src.wmac_perceptive.profiling import ProfileWindow,observe_training
    from src.wmac_perceptive.runner import PerceptiveRunner
    window=ProfileWindow(options.profile_wait_iterations,options.profile_active_iterations,options.profile_memory)
    cfg=parse_config(training_arguments)
    with observe_training(PerceptiveRunner,window):train(cfg)


if __name__=="__main__":main()
