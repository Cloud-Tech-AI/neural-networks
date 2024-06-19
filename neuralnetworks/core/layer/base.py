import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from ...helpers.activation_functions.activation import Activation, Sigmoid
from ...helpers.normalize_functions.normalizer import Normalizer, ZScore


@dataclass
class Layer:
    input_size: int
    output_size: int
    input: np.ndarray = field(default_factory=lambda: np.array([]))
    output: np.ndarray = field(default_factory=lambda: np.array([]))

    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    bias: np.ndarray = field(default_factory=lambda: np.array([]))
    grad_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    grad_bias: np.ndarray = field(default_factory=lambda: np.array([]))
    grad_pre_activation: np.ndarray = field(default_factory=lambda: np.array([]))
    grad_current_layer: np.ndarray = field(default_factory=lambda: np.array([]))

    activation: Activation = Sigmoid()
    normalizer: Normalizer = ZScore()
    type: Optional[Literal["dense"]] = None

    def initialize_weights(self):
        self.normalizer.normalize(np.random.rand(self.output_size, self.input_size))
        self.weights = self.normalizer.output
        self.normalizer.normalize(np.random.rand(self.output_size, 1))
        self.bias = self.normalizer.output

    def reset_gradients(self):
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
