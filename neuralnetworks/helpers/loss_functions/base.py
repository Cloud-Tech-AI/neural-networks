from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Loss:
    type: Optional[
        Literal["squared_error", "cross_entropy", "binary_cross_entropy"]
    ] = None
    output: float = None
    step_loss: float = 0
    epoch_loss: float = 0

    def reset_step_loss(self):
        self.step_loss = 0

    def reset_epoch_loss(self):
        self.epoch_loss = 0
