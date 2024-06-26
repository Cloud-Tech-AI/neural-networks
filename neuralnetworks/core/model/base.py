from dataclasses import dataclass, field
from typing import Optional

from ..layer.base import Layer
from ...helpers.loss_functions.loss import Loss, SquaredError


@dataclass
class Model:
    layers: list[Layer] = field(default_factory=lambda: [])
    loss: Loss = SquaredError()
    optimizer: Optional[str] = None
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32

    def add(self, layer):
        self.layers.append(layer)

    def get_batches(self, x_train, y_train):
        for i in range(0, len(x_train), self.batch_size):
            yield x_train[i : i + self.batch_size], y_train[i : i + self.batch_size]

    def forward(self, x):
        self.layers[0].forward(x)
        for idx, layer in enumerate(self.layers[1:], 1):
            layer.forward(self.layers[idx - 1].output)

    def get_loss(self, y):
        self.loss.get_loss(y, self.layers[-1].output)

    def backward(self, y):
        self.layers[-1].backward(y, loss=self.loss.type)
        for idx, layer in enumerate(reversed(self.layers[:-1])):
            layer.backward(self.layers[-1 - idx].grad_current_layer)

    def train(self, x_train, y_train):
        curr_epoch = 0
        while curr_epoch < self.epochs:
            step = 0
            for x_batch, y_batch in self.get_batches(x_train, y_train):
                for x, y in zip(x_batch, y_batch):
                    self.forward(x)
                    self.get_loss(y)
                    self.backward(y)
                for layer in self.layers:
                    layer.weights -= self.learning_rate * layer.grad_weights
                    layer.bias -= self.learning_rate * layer.grad_bias
                    layer.reset_gradients()
                print(
                    f"Epoch: {curr_epoch} Step: {step} Loss: {self.loss.step_loss/len(x_batch)}"
                )
                self.loss.reset_step_loss()
                step += 1
            print(f"Epoch: {curr_epoch} Avg Loss: {self.loss.epoch_loss/len(x_train)}")
            self.loss.reset_epoch_loss()
            curr_epoch += 1
