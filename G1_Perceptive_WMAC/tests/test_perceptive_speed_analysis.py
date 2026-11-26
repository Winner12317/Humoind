"""Read-only speed analysis must amortize intermittent AMP and skip startup."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_wmac_perceptive_speed import analyze,STAGES


class SpeedAnalysisTest(unittest.TestCase):
    def test_resumed_run_excludes_initial_rows_and_averages_sparse_amp_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'metrics.csv'
            fields=('iteration','wall_seconds','sim_steps','imagination_steps',*STAGES,
                    'discriminator_updates_applied','amp_policy_exposures')
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for index,wall in enumerate((100.,200.,500.,504.,508.,512.)):
                    row={name:0 for name in fields}
                    row.update(iteration=1000+index,wall_seconds=wall,sim_steps=index*10240,
                        imagination_steps=index*131072,collect_seconds=.4,model_seconds=1.4,
                        actor_critic_seconds=1.)
                    if index==4:
                        row.update(disc_seconds=3.6,discriminator_updates_applied=20,
                                   amp_policy_exposures=491520)
                    writer.writerow(row)
                stream.write('1006,')  # Active writer has not completed this row.
            (root/'source.json').write_text(json.dumps({'commit':'931dee2a'}))
            (root/'config.json').write_text(json.dumps({'num_envs':4096,'hiking_training_version':2}))
            before=path.read_bytes()
            result=analyze(root,window=3,skip_first=2)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(result['recorded_source_commit'],'931dee2a')
            self.assertEqual(result['interval']['start_iteration'],1002)
            self.assertEqual(result['interval']['end_iteration'],1005)
            self.assertEqual(result['interval']['seconds_per_iteration'],4.)
            self.assertEqual(result['interval']['real_transitions_per_second'],2560.)
            self.assertEqual(result['interval']['imagined_transitions_per_second'],32768.)
            self.assertEqual(result['AMP']['rows_with_updates'],1)
            self.assertEqual(result['AMP']['optimizer_updates'],20.)
            self.assertAlmostEqual(result['AMP']['amortized_seconds_per_record'],1.2)
            self.assertEqual(result['AMP']['mean_seconds_on_update_rows'],3.6)
            self.assertEqual(result['incomplete_or_nonfinite_rows_discarded'],1)
            self.assertIsNone(result['completed_runtime_graph_flags'])

    def test_missing_timers_or_insufficient_steady_data_are_explicit_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metrics.csv'
            path.write_text('iteration,wall_seconds\n1,10\n')
            with self.assertRaisesRegex(ValueError,'Missing timer fields'):analyze(path)
            with self.assertRaisesRegex(ValueError,'window must be positive'):analyze(path,window=0)

    def test_old_logs_without_amp_counters_remain_unknown_instead_of_zero_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metrics.csv'
            fields=('iteration','wall_seconds','sim_steps','imagination_steps',*STAGES)
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for iteration in (1,2):
                    row={key:0 for key in fields}
                    row.update(iteration=iteration,wall_seconds=iteration*4,
                        sim_steps=iteration*10240,imagination_steps=iteration*131072,
                        collect_seconds=.5,model_seconds=1.5,disc_seconds=.2,actor_critic_seconds=1.8)
                    writer.writerow(row)
            result=analyze(path,window=1,skip_first=0)
            for key in ('optimizer_updates','policy_window_exposures','rows_with_updates','rows_without_updates'):
                self.assertIsNone(result['AMP'][key])
            self.assertEqual(result['AMP']['amortized_seconds_per_record'],.2)


if __name__=='__main__':unittest.main()
