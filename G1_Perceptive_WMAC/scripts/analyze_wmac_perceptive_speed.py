"""Read an existing metrics CSV; never start training or use a GPU.

The stage timers are approximate host wall time. Actor/Critic time also
contains validation. Intermittent AMP cost is averaged over every iteration,
not the median of mostly-zero discriminator rows.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
STAGES=('collect_seconds','model_seconds','disc_seconds','actor_critic_seconds',
        'save_seconds','evaluation_seconds')
GRAPH_FLAGS=('wm_graph_captured','actor_graph_captured','critic_graph_captured','amp_gp_graph_captured',
             'amp_logical_graph_captured','amp_micro_graph_captured')


def number(value):
    try:value=float(value)
    except (TypeError,ValueError):return None
    return value if math.isfinite(value) else None


def optional_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def select_run(path=None):
    if path is not None:
        path=Path(path).expanduser().resolve()
        if path.is_file():return path
        if (path/'metrics.csv').is_file():return path/'metrics.csv'
        raise ValueError(f'No metrics.csv in {path}')
    paths=list((ROOT/'logs/wmac_perceptive').glob('*/metrics.csv'))
    if not paths:raise ValueError('No perceptive metrics found; specify --run /path/to/run')
    return max(paths,key=lambda value:value.stat().st_mtime)


def analyze(path,*,window=300,skip_first=50):
    if window<1 or skip_first<0:raise ValueError('window must be positive; skip-first must be nonnegative')
    path=select_run(path)
    recent=deque(maxlen=window+1)
    valid_rows=discarded=0
    required=('iteration','wall_seconds','sim_steps','imagination_steps',*STAGES[:4])
    selected_fields=set(required)|set(STAGES)|set(GRAPH_FLAGS)|{'discriminator_updates_applied','amp_policy_exposures'}
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream)
        missing=set(required).difference(reader.fieldnames or ())
        if missing:raise ValueError(f'Missing timer fields: {sorted(missing)}')
        for record in reader:
            row={key:number(record.get(key)) for key in selected_fields}
            if any(row.get(key) is None for key in required):
                discarded+=1;continue  # A live writer can have an incomplete last row.
            valid_rows+=1
            if valid_rows>skip_first:recent.append(row)
    if len(recent)<2:
        raise ValueError(f'Need at least2 complete rows after excluding first{skip_first}; found{len(recent)}')
    rows=list(recent);first,last=rows[0],rows[-1];interval=rows[1:]
    wall=last['wall_seconds']-first['wall_seconds']
    iterations=last['iteration']-first['iteration']
    if wall<=0 or iterations<=0:raise ValueError('Metrics interval must advance wall time and iterations')
    totals={key:sum(row.get(key) or 0. for row in interval) for key in STAGES}
    measured=sum(totals.values())
    stages={key:{'mean_seconds_per_record':value/len(interval),
                 'total_seconds':value,'percent_of_measured_stages':100*value/measured if measured else None}
            for key,value in totals.items()}
    known_updates=all(row.get('discriminator_updates_applied') is not None for row in interval)
    known_exposures=all(row.get('amp_policy_exposures') is not None for row in interval)
    active=([row for row in interval if row['discriminator_updates_applied']>0] if known_updates else [])
    config=optional_json(path.parent/'config.json')
    source=optional_json(path.parent/'source.json')
    runtime=optional_json(path.parent/'runtime.json')
    return dict(run_directory=str(path.parent.resolve()),metrics=str(path),
        recorded_source_commit=source.get('commit'),recorded_source_status=source.get('status'),
        selected_complete_rows=len(rows),incomplete_or_nonfinite_rows_discarded=discarded,
        initial_rows_excluded=skip_first,
        interval={'start_iteration':int(first['iteration']),'end_iteration':int(last['iteration']),
            'iterations':int(iterations),'wall_seconds':wall,'seconds_per_iteration':wall/iterations,
            'real_transitions_per_second':(last['sim_steps']-first['sim_steps'])/wall,
            'imagined_transitions_per_second':(last['imagination_steps']-first['imagination_steps'])/wall,
            'iterations_per_hour':iterations/wall*3600},
        stages=stages,
        AMP={'rows_with_updates':len(active) if known_updates else None,
            'rows_without_updates':len(interval)-len(active) if known_updates else None,
            'optimizer_updates':sum(row['discriminator_updates_applied'] for row in interval) if known_updates else None,
            'mean_seconds_on_update_rows':statistics.mean(row['disc_seconds'] for row in active) if active else None,
            'amortized_seconds_per_record':stages['disc_seconds']['mean_seconds_per_record'],
            'policy_window_exposures':sum(row['amp_policy_exposures'] for row in interval) if known_exposures else None},
        recorded_configuration={key:config.get(key) for key in ('hiking_training_version','num_envs',
            'wm_batch_size','wm_updates','wm_window','actor_batch_size','imagination_horizon',
            'precision','compile','cuda_graphs','benchmark_sync')},
        live_CSV_graph_flags={key:last.get(key) for key in GRAPH_FLAGS},
        completed_runtime_graph_flags={key:runtime.get(key) for key in (
            'wm_full_update_graph_captured','actor_full_update_graph_captured',
            'critic_full_update_graph_captured','amp_gp_graph_captured',
            'wm_update_granularity','policy_update_granularity','wm_shared_reconstruction',
            'amp_execution_granularity')} if runtime else None,
        notes=['Stage percentages use summed measured phase times, not GPU utilization or kernel time.',
            'actor_critic_seconds includes validation. Save/evaluation costs remain in the report.',
            'Throughputs use the complete training wall interval, not simulator-only speed.',
            'Runtime graph flags are unavailable until runtime.json is written; configured cuda_graphs is not proof of capture.',
            'Live graph flags describe recorded graphs, not execution of every stage on every row; AMP is intermittent.',
            'Initial exclusion counts rows of this run, including resumed runs, not absolute iteration numbers.'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,help='Run directory or metrics.csv; default most recently modified perceptive CSV')
    parser.add_argument('--window',type=int,default=300,help='Number of recent logged intervals')
    parser.add_argument('--skip-first',type=int,default=50,help='Exclude initial complete rows of this run')
    args=parser.parse_args(argv)
    try:result=analyze(args.run,window=args.window,skip_first=args.skip_first)
    except (ValueError,OSError,json.JSONDecodeError) as error:parser.error(str(error))
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
