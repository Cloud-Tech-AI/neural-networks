import numpy as np
from dataclasses import dataclass, field
from helpers.activation_functions.activation import Activation, Sigmoid


@dataclass
class Layer:
    input_size: int
    output_size: int
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    bias: np.ndarray = field(default_factory=lambda: np.array([]))
    activation: Activation = Sigmoid()
    type: str = None
    normalizer: str = 'z-score'
    input: np.ndarray = field(default_factory=lambda: np.array([]))
    output: np.ndarray = field(default_factory=lambda: np.array([]))
    grad_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    grad_bias: np.ndarray = field(default_factory=lambda: np.array([]))
    grad_pre_activation: np.ndarray = field(default_factory=lambda: np.array([]))
    grad_current_layer: np.ndarray = field(default_factory=lambda: np.array([]))

    def initialize_weights(self):
        self.weights = np.random.rand(self.output_size, self.input_size)
        self.bias = np.random.rand(self.output_size, 1)
        # call normalizer
        
    def reset_gradients(self):
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

@dataclass
class Dense(Layer):
    def __post_init__(self):
        self.type = 'dense'
        np.random.seed(42)
        self.initialize_weights()
        self.reset_gradients()

    def forward(self, input):
        self.input = input
        pre_activation = np.dot(self.weights, self.input) + self.bias
        self.activation.activate(pre_activation)
        self.output = self.activation.output

    def backward(self, grad_prev_layer, loss = None):
        if loss:
            if self.activation.type == 'softmax' and loss == 'cross_entropy':
                self.grad_pre_activation = -(grad_prev_layer - self.output)
                self.grad_weights += np.dot(self.grad_pre_activation,self.input.T)
                self.grad_bias += self.grad_pre_activation
                self.grad_current_layer = np.dot(self.weights.T, self.grad_pre_activation)
            elif self.layers[-1].activation.type == 'sigmoid' and self.loss.type == 'binary_cross_entropy':
                raise Exception('Combination of sigmoid activation and binary cross entropy loss is not supported.')
            elif self.layers[-1].activation.type == 'sigmoid' and self.loss.type == 'squared_error':
                raise Exception('Combination of sigmoid activation and squared error loss is not supported.')
        else:
            if self.activation.type == 'sigmoid':
                self.grad_pre_activation = grad_prev_layer * self.output * (1 - self.output)
                self.grad_weights += np.dot(self.grad_pre_activation,self.input.T)
                self.grad_bias += self.grad_pre_activation
                self.grad_current_layer = np.dot(self.weights.T, self.grad_pre_activation)
            else:
                raise Exception('Only sigmoid activation is supported for now.')
