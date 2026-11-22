"""Check that compiled reward and backward see live EMA buffers without recompiling."""
from pathlib import Path
import json
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from torch._dynamo.testing import CompileCounterWithBackend

from src.wmac.amp_rewards import CenteredAmpReward


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    reward = CenteredAmpReward(decay=.99).to(device)
    counter = CompileCounterWithBackend('inductor')
    compiled = torch.compile(reward, backend=counter, fullgraph=True)
    score = torch.linspace(-4., 4., 4096, device=device, requires_grad=True)
    rows = []
    for robot_batch_mean in (-.5, .8, -.8):
        reward.update(score.new_tensor([robot_batch_mean]), score.new_tensor([.9]))
        actual = compiled(score)
        expected = reward(score)
        actual_grad, = torch.autograd.grad(actual.sum(), score)
        expected_grad, = torch.autograd.grad(expected.sum(), score)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-8)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-8)
        assert torch.isfinite(actual_grad).all()
        rows.append(dict(robot_mean=float(reward.robot_mean), mean_reward=float(actual.mean()),
                         max_forward_error=float((actual-expected).abs().max()),
                         max_backward_error=float((actual_grad-expected_grad).abs().max())))
    assert counter.frame_count == 1, counter.frame_count
    assert len({row['mean_reward'] for row in rows}) == len(rows)
    result = dict(device=str(device), torch=torch.__version__, fullgraph=True,
                  compiled_frames=counter.frame_count, checks=rows,
                  physical_steps=0, optimizer_steps=0)
    destination = REPO/'docs/validation/centered_amp_compile.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
