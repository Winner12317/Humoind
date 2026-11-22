# Third-party notices

The `supermimic_official` compatibility profile references the official
[HybridRobotics/SuperMimic](https://github.com/HybridRobotics/supermimic), pinned
to `abc124c1bd5976341034668af3d12c087f2f1f55`. Copyright 2026 The SuperMimic
Authors; the Apache-2.0 notice is retained in
[licenses/supermimic-Apache-2.0.txt](licenses/supermimic-Apache-2.0.txt).

The local Muon compatibility implementation in `src/wmac/optimizers.py` follows
PyTorch v2.10.0's Muon math for the existing PyTorch 2.7 environment. Its
BSD-3-Clause notice is retained in
[licenses/pytorch-v2.10-BSD-3-Clause.txt](licenses/pytorch-v2.10-BSD-3-Clause.txt).
Reference optimizer grouping follows the official SuperMimic learner utilities.
No official tracking assets, perceptive framework or terrain datasets are
included by this flat-task alignment. The original local-residual model and
AMP/recovery extensions remain the implementation described below.

The local-increment dynamics parameterization and training-loop design reference
[SuperMimic's unofficial reproduction](https://github.com/zita-ch/supermimic_vibed),
commit `0525a23a2dcfa61f8543f98b84c075fe66d9d286`. Its MIT license follows.
Existing mjlab and RSL-RL components retain their own license headers.

MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## InstantAMP centered reward adaptation

The fixed positive-scale robot-score-centered tanh reward and real-score EMA
semantics in `src/wmac/amp_rewards.py` are adapted from
RoboticsResearchHubVN/InstantAMP commit
`9fc8031055a18d58e84d9de90639660e7aa6b491`,
`instantamp/amp.py:106-119,138-148`. The original project uses WGAN; this
repository retains its LSGAN discriminator and defines scale in LSGAN units.
MIT notice: `licenses/InstantAMP-MIT.txt`. Copyright (c) 2026 InstantAMP
contributors; Copyright (c) 2026 Nico Bohlinger (XSoma).

## HoST and HumanUP reward concepts

`src/wmac/smooth_rewards.py` adapts the mathematical reward concepts from
InternRobotics/HoST commit `70bb580949a336a920833700e4b5dc3bf7fe87ce`
(second action difference; project MIT) and RunpeiDong/humanup commit
`7516e0f27e6f4d1e7365cf64ea577a78247bd8cb`
(joint velocity, effort and power; relevant source BSD-3-Clause).
This implementation uses the current repository's analytic Unitree PD command
for imagined costs, with pre-clipping request norm as an explicit BPTT adaptation.
It does not reproduce either project's controller, curriculum or training stages.
Source URLs and transfer details are in `docs/SMOOTH_REWARD_ADAPTATION_zh.md`.

License notices: `licenses/HoST-MIT.txt`, `licenses/HumanUP-BSD-3-Clause.txt`.
