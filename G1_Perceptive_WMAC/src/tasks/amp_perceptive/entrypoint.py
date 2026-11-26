"""Prevent the legacy PPO entrypoint from silently dropping the camera."""


class PerceptiveTrainingEntrypointRequired:
    """Registry guard; the dedicated WMAC entrypoint constructs its own runner."""

    def __init__(self,*args,**kwargs):
        raise ValueError(
            "Unitree-G1-AMP-Perceptive-Hiking requires the depth-aware WMAC runner. "
            "Start it with python scripts/train_wmac_perceptive.py. "
            "The legacy scripts/train.py PPO runner only reads the "
            "proprioceptive Actor observation and would ignore perception."
        )
