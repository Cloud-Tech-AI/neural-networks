from dataclasses import Field, dataclass

from ...core.layer.layer import Layer
from ...helpers.loss_functions.loss import Loss, SquaredError

@dataclass
class Model:
    layers: list[Layer] = Field.default_factory(list)
    loss: Loss = SquaredError()
    optimizer: str = None
    learning_rate: float = 0.01
    epochs: int = 1000
    batch_size: int = 32

    def add(self, layer):
        self.layers.append(layer)

    def get_batches(self, X_train, y_train):
        for i in range(0, len(X_train), self.batch_size):
            yield X_train[i:i + self.batch_size], y_train[i:i + self.batch_size]

    def forward(self, X_batch):
        self.layers[0].forward(X_batch)
        for idx, layer in enumerate(self.layers[1:]):
            layer.forward(layer[idx].output)
    
    def get_loss(self, y_batch):
        self.loss.get_loss(y_batch, self.layers[-1].output)
    
    def backward(self):
        if self.layers[-1].activation.type == 'softmax' and self.loss.type == 'cross_entropy':
            pass
        elif self.layers[-1].activation.type == 'sigmoid' and self.loss.type == 'binary_cross_entropy':
            raise Exception('Combination of sigmoid activation and binary cross entropy loss is not supported.')
        elif self.layers[-1].activation.type == 'sigmoid' and self.loss.type == 'squared_error':
            raise Exception('Combination of sigmoid activation and squared error loss is not supported.')
        
        
        for idx, layer in enumerate(self.layers[::-1]):
            layer.backward(self.loss.output, self.learning_rate)

    def train(self, X_train, y_train):
        curr_epoch = 0
        while curr_epoch < self.epochs:
            for X_batch, y_batch in self.get_batches(X_train, y_train):
                self.forward(X_batch)
                self.get_loss(y_batch)
                self.backward()
            
            # update weights
            curr_epoch += 1
