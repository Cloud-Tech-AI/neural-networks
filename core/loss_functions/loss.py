from dataclasses import dataclass
import numpy as np
from pydantic import Field

@dataclass
class Loss:
    y_true: np.ndarray = Field.default_factory(np.array([]))
    y_pred: np.ndarray = Field.default_factory(np.array([]))

    def squared_error(self):
        """
        Squared error loss
        """
        return np.mean(np.power(self.y_true - self.y_pred, 2))
    
    def cross_entropy(self):
        """
        Cross entropy loss
        """
        return -np.sum(self.y_true * np.log(self.y_pred))
    
    def binary_cross_entropy(self):
        """
        Binary cross entropy loss
        """
        return -np.mean(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))