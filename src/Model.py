import keras.models
from keras.layers import *


class Baseline(keras.models.Sequential):
    def __init__(self):
        super(Baseline, self).__init__()
        self.add(InputLayer(input_shape=[64, 64, 1]))
        self.add(Conv2D(filters=16, kernel_size=11, strides=4, padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=3, strides=2, padding="same"))

        self.add(Conv2D(filters=20, kernel_size=5, strides=1, padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=3, strides=2, padding="same"))

        self.add(Conv2D(filters=30, kernel_size=3, strides=1, padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=3, strides=2, padding="same"))

        self.add(Flatten())
        self.add(Dense(48, activation="relu"))
        self.add(Dense(2, activation="softmax"))
