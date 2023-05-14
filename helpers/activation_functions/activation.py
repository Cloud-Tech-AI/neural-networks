import numpy as np
from pydantic import Field, dataclass

@dataclass
class Activation:
    input: np.ndarray = Field.default_factory(np.array([]))
    type: str = None
    output: np.ndarray = Field.default_factory(np.array([]))

class Sigmoid(Activation):
    def __post_init__(self):
        self.type = 'sigmoid'

    def activate(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-self.input))
    
class ReLU(Activation):
    def __post_init__(self):
        self.type = 'relu'

    def relu(self, input):
        self.input = input
        self.output = np.maximum(0, self.input)

class Tanh(Activation):
    def __post_init__(self):
        self.type = 'tanh'
    
    def tanh(self, input):
        self.input = input
        self.output = (np.exp(self.input) - np.exp(-self.input)) / (np.exp(self.input) + np.exp(-self.input))
    
class Softmax(Activation):
    def __post_init__(self):
        self.type = 'softmax'

    def softmax(self, input):
        self.input = input
        self.output = np.exp(self.input) / np.sum(np.exp(self.input))