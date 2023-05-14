import numpy as np
from pydantic import Field, dataclass
from ...helpers.activation_functions.activation import Activation, Sigmoid


@dataclass
class Layer:
    input_size: int
    output_size: int
    weights: np.ndarray = Field.default_factory(np.array([]))
    bias: np.ndarray = Field.default_factory(np.array([]))
    activation: Activation = Sigmoid()
    type: str = None
    normalizer: str = 'z-score'
    input: np.ndarray = Field.default_factory(np.array([]))
    output: np.ndarray = Field.default_factory(np.array([]))
    grad_weights: np.ndarray = Field.default_factory(np.array([]))
    grad_bias: np.ndarray = Field.default_factory(np.array([]))

    def initialize_weights(self):
        self.weights = np.random.rand(self.input_size, self.output_size)
        self.bias = np.random.rand(self.output_size)
        # call normalizer
        
    def reset_gradients(self):
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

@dataclass
class Dense(Layer):
    def __post_init__(self):
        self.type = 'dense'
        self.initialize_weights()
        self.reset_gradients()

    def forward(self, input):
        self.input = input
        pre_activation = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation.activate(pre_activation)
    
    def backward(self, output_error, learning_rate):
        pass
