import numpy as np
from pydantic import dataclass

@dataclass
class Loss:
    type: str = None
    output: float = None

class SquaredError(Loss):
    def __init__(self):
        super().__init__(type='squared_error')

    def get_loss(self, y_true, y_pred):
        self.output = np.power(y_true - y_pred, 2)
    
class CrossEntropy(Loss):
    def __init__(self):
        super().__init__(type='cross_entropy')

    def get_loss(self, y_true, y_pred):
        self.output = -np.sum(y_true * np.log(y_pred))

class BinaryCrossEntropy(Loss):
    def __init__(self):
        super().__init__(type='binary_cross_entropy')

    def get_loss(self, y_true, y_pred):
        self.output = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))