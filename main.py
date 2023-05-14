from .helpers.loss_functions.loss import CrossEntropy
from .core.model.model import Model
from .helpers.activation_functions.activation import ReLU, Softmax
from .core.layer.layer import Dense

# create a model
model = Model(loss=CrossEntropy())
model.add(Dense(input_size=2, output_size=3))
model.add(Dense(input_size=3, output_size=5))
model.add(Dense(input_size=5, output_size=1, activation=Softmax()))

# train the model
model.train(X_train, y_train)