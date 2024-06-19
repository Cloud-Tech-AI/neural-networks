import numpy as np
from dataclasses import dataclass

from .base import Normalizer


@dataclass
class MinMax(Normalizer):
    def __post_init__(self):
        self.type = "min-max"

    def normalize(self, input, min=0, max=1):
        self.input = input
        self.output = ((self.input - min / (max - min)) * (max - min)) + min


@dataclass
class ZScore(Normalizer):
    def __post_init__(self):
        self.type = "z-score"

    def normalize(self, input):
        self.input = input
        self.output = (self.input - np.mean(self.input)) / np.std(self.input)
