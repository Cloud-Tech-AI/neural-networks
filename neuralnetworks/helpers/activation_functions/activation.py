import numpy as np
from dataclasses import dataclass

from .base import Activation


@dataclass
class Sigmoid(Activation):
    def __post_init__(self):
        self.type = "sigmoid"

    def activate(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-self.input))


@dataclass
class ReLU(Activation):
    def __post_init__(self):
        self.type = "relu"

    def activate(self, input):
        self.input = input
        self.output = np.maximum(0, self.input)


@dataclass
class Tanh(Activation):
    def __post_init__(self):
        self.type = "tanh"

    def activate(self, input):
        self.input = input
        self.output = (np.exp(self.input) - np.exp(-self.input)) / (
            np.exp(self.input) + np.exp(-self.input)
        )


@dataclass
class Softmax(Activation):
    def __post_init__(self):
        self.type = "softmax"

    def activate(self, input):
        self.input = input
        self.output = np.exp(self.input) / np.sum(np.exp(self.input))
