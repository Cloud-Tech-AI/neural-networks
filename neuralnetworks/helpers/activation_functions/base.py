import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class Activation:
    input: np.ndarray = field(default_factory=lambda: np.array([]))
    type: Optional[Literal["sigmoid", "relu", "tanh", "softmax"]] = None
    output: np.ndarray = field(default_factory=lambda: np.array([]))
