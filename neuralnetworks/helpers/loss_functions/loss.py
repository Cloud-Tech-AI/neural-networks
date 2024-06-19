import numpy as np
from dataclasses import dataclass

from .base import Loss


@dataclass
class SquaredError(Loss):
    def __post_init__(self):
        self.type = "squared_error"

    def get_loss(self, y_true, y_pred):
        self.output = np.power(y_true - y_pred, 2)
        self.step_loss += self.output
        self.epoch_loss += self.output


@dataclass
class CrossEntropy(Loss):
    def __post_init__(self):
        self.type = "cross_entropy"

    def get_loss(self, y_true, y_pred):
        self.output = -np.sum(y_true * np.log(y_pred))
        self.step_loss += self.output
        self.epoch_loss += self.output


@dataclass
class BinaryCrossEntropy(Loss):
    def __post_init__(self):
        self.type = "binary_cross_entropy"

    def get_loss(self, y_true, y_pred):
        self.output = -np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        self.step_loss += self.output
        self.epoch_loss += self.output
