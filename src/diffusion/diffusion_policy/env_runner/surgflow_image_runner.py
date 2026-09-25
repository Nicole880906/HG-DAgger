"""Offline runner used until closed-loop dVRK evaluation is implemented."""

from typing import Dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class SurgFlowImageRunner(BaseImageRunner):
    def run(self, policy: BaseImagePolicy) -> Dict:
        # Validation loss and held-out action error are computed by the
        # workspace. Robot rollouts must be added separately with safety
        # interlocks, so this runner intentionally performs no actuation.
        return {}

