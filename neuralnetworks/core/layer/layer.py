import numpy as np
from dataclasses import dataclass

from .base import Layer


@dataclass
class Dense(Layer):
    def __post_init__(self):
        np.random.seed(42)
        self.type = "dense"
        self.initialize_weights()
        self.reset_gradients()

    def forward(self, input):
        self.input = input
        pre_activation = np.dot(self.weights, self.input) + self.bias
        self.activation.activate(pre_activation)
        self.output = self.activation.output

    def backward(self, grad_prev_layer, loss=None):
        if loss:
            # Final Layer
            if self.activation.type == "softmax" and loss == "cross_entropy":
                self.grad_pre_activation = -(grad_prev_layer - self.output)
            elif self.activation.type == "sigmoid" and loss == "squared_error":
                self.grad_pre_activation = (
                    -(grad_prev_layer - self.output) * self.output * (1 - self.output)
                )
            else:
                raise Exception(
                    f"Activation {self.activation.type} and Loss {loss} not supported for final layer."
                )
        else:
            # Intermediate Layer
            if self.activation.type == "sigmoid":
                self.grad_pre_activation = (
                    grad_prev_layer * self.output * (1 - self.output)
                )
            else:
                raise Exception(
                    f"Activation {self.activation.type} not supported for intermediate layers."
                )

        self.grad_weights += np.dot(self.grad_pre_activation, self.input.T)
        self.grad_bias += self.grad_pre_activation
        self.grad_current_layer = np.dot(self.weights.T, self.grad_pre_activation)
