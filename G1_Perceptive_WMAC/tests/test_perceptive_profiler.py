"""The opt-in profiler observes real loop calls without changing training."""
from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import torch

from scripts.profile_wmac_perceptive import arguments
from src.wmac_perceptive.profiling import ProfileWindow,observe_training


class TinyLoop:
    def __init__(self,directory):
        self.cfg=SimpleNamespace(device="cpu",num_envs=4096)
        self.log_dir=Path(directory)
        self.iteration=0
        self.calls=[]
        self.weight=torch.tensor([1.,2.],requires_grad=True)
        self.opt=torch.optim.Adam((self.weight,),lr=1e-3)
        self.draws=[]

    def collect(self,count):
        self.calls.append(("collect",count))
        self.draws.append((torch.rand(2),random.random()))

    def fit_model(self):
        self.calls.append(("fit_model",))
        self.opt.zero_grad(set_to_none=True)
        loss=(self.weight-self.draws[-1][0]).square().mean()
        loss.backward();self.opt.step()

    def update_discriminator(self):self.calls.append(("update_discriminator",))
    def validate_model(self):self.calls.append(("validate_model",))
    def update_actor_critic(self):self.calls.append(("update_actor_critic",))
    def save(self,path):self.calls.append(("save",str(path)))
    def _log(self,row):self.calls.append(("log",row["iteration"]))

    def learn(self):
        self.collect(8)
        for index in range(6):
            self.collect(2)
            self.fit_model()
            self.update_discriminator()
            if index%2==0:self.validate_model()
            self.update_actor_critic()
            self.iteration=index+1
            if self.iteration%3==0:self.save("checkpoint")
            self._log({"iteration":self.iteration})


class ProfilerTest(unittest.TestCase):
    def test_one_bounded_trace_preserves_config_calls_rng_weights_moments_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            reference=TinyLoop(Path(directory)/"reference")
            observed=TinyLoop(Path(directory)/"observed")
            torch.manual_seed(44);random.seed(44)
            reference.learn()
            torch_state,python_state=torch.get_rng_state(),random.getstate()
            torch.manual_seed(44);random.seed(44)
            methods={name:getattr(TinyLoop,name) for name in ("learn","_log","collect","fit_model","save")}
            before=vars(observed.cfg).copy()
            with observe_training(TinyLoop,ProfileWindow(wait_iterations=1,active_iterations=2)):
                observed.learn()
            self.assertEqual(before,vars(observed.cfg))
            self.assertEqual(observed.calls,reference.calls)
            self.assertEqual(observed.iteration,6)
            torch.testing.assert_close(torch.get_rng_state(),torch_state,rtol=0,atol=0)
            self.assertEqual(random.getstate(),python_state)
            torch.testing.assert_close(observed.weight,reference.weight,rtol=0,atol=0)
            for key,value in observed.opt.state[observed.weight].items():
                torch.testing.assert_close(value,reference.opt.state[reference.weight][key],rtol=0,atol=0)
            for name,method in methods.items():self.assertIs(getattr(TinyLoop,name),method)
            summary=json.loads((observed.log_dir/"profile/summary.json").read_text())
            self.assertEqual(summary["recorded_complete_iterations"],2)
            self.assertTrue(summary["full_window_completed"])
            self.assertEqual(summary["iteration_at_trace_end"],4)
            self.assertFalse(summary["training_configuration_modified"])
            trace=json.loads((observed.log_dir/"profile/trace.json").read_text())
            self.assertIn("WMAC::fit_model",{event.get("name") for event in trace["traceEvents"]})
            self.assertTrue((observed.log_dir/"profile/operators.txt").is_file())

    def test_exception_restores_inherited_methods_without_adding_saved_updates(self):
        class FailingLoop(TinyLoop):
            def fit_model(self):raise RuntimeError("fixture failure")
        inherited=FailingLoop.learn
        with tempfile.TemporaryDirectory() as directory:
            runner=FailingLoop(directory)
            with self.assertRaisesRegex(RuntimeError,"fixture failure"):
                with observe_training(FailingLoop,ProfileWindow(wait_iterations=100)):
                    runner.learn()
            self.assertIs(FailingLoop.learn,inherited)
            self.assertNotIn("learn",FailingLoop.__dict__)
            self.assertNotIn("collect",FailingLoop.__dict__)
            self.assertFalse((Path(directory)/"profile").exists())
            self.assertEqual(runner.iteration,0)
            self.assertFalse(runner.opt.state)

    def test_cli_forwards_training_arguments_verbatim_and_validates_observer_window(self):
        training=["--num-envs","4096","--iterations","70000","--save-interval","1000","--run-name","stones"]
        options,forwarded=arguments(["--profile-wait-iterations","75","--",*training])
        self.assertEqual(forwarded,training)
        self.assertEqual(options.profile_wait_iterations,75)
        self.assertEqual(options.profile_active_iterations,24)
        for kw in ({"wait_iterations":-1},{"active_iterations":0},{"active_iterations":True}):
            with self.assertRaises(ValueError):ProfileWindow(**kw)


if __name__=="__main__":unittest.main()
