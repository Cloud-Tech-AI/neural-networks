import numpy as np
from dataclasses import dataclass, field


@dataclass
class Normalizer:
    input: np.ndarray = field(default_factory=lambda: np.array([]))
    type: str = None
    output: np.ndarray = field(default_factory=lambda: np.array([]))
