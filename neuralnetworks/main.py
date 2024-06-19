import pandas as pd
import numpy as np

from .helpers.loss_functions.loss import CrossEntropy
from .core.model.base import Model
from .helpers.activation_functions.activation import Softmax
from .core.layer.layer import Dense


# prepare data
df = pd.read_csv(
    "/home/ishan/vscode-workspace/neural-networks/neuralnetworks/data/seeds_dataset.csv"
)
x_train = []
y_train = []
for idx in range(len(df)):
    x_arr = np.asarray(df.loc[idx][:-1])
    y_arr = np.asarray([1, 0]) if df.loc[idx][0] == 1 else np.asarray([0, 1])

    x_train.append(np.expand_dims(x_arr, axis=1))
    y_train.append(np.expand_dims(y_arr, axis=1))

# create a model
model = Model(loss=CrossEntropy())
model.add(Dense(input_size=len(x_train[0]), output_size=3))
model.add(Dense(input_size=3, output_size=5))
model.add(Dense(input_size=5, output_size=len(y_train[0]), activation=Softmax()))

# train the model
model.train(x_train, y_train)
