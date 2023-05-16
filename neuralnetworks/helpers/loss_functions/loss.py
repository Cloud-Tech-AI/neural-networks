import numpy as np
from dataclasses import dataclass

@dataclass
class Loss:
    type: str = None
    output: float = None
    step_loss: float = 0
    epoch_loss: float = 0

    def reset_step_loss(self):
        self.step_loss = 0
    
    def reset_epoch_loss(self):
        self.epoch_loss = 0

class SquaredError(Loss):
    def __init__(self):
        super().__init__(type='squared_error')

    def get_loss(self, y_true, y_pred):
        self.output = np.power(y_true - y_pred, 2)
        self.step_loss += self.output
        self.epoch_loss += self.output
    
class CrossEntropy(Loss):
    def __init__(self):
        super().__init__(type='cross_entropy')

    def get_loss(self, y_true, y_pred):
        self.output = -np.sum(y_true * np.log(y_pred))
        self.step_loss += self.output
        self.epoch_loss += self.output

class BinaryCrossEntropy(Loss):
    def __init__(self):
        super().__init__(type='binary_cross_entropy')

    def get_loss(self, y_true, y_pred):
        self.output = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        self.step_loss += self.output
        self.epoch_loss += self.output