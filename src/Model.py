import keras.models
from keras.layers import *


class Baseline(keras.models.Sequential):
    def __init__(self):
        super(Baseline, self).__init__()
        self.add(InputLayer(input_shape=[64, 64, 1]))
        self.add(Conv2D(filters=32, kernel_size=5, strides=1, padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=5, padding="same"))

        self.add(Conv2D(filters=50, kernel_size=5, strides=1, padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=5, padding="same"))

        self.add(Conv2D(filters=80, kernel_size=5, strides=1, padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=5, padding="same"))

        self.add(Dropout(0.25))
        self.add(Flatten())
        self.add(Dense(512, activation="relu"))
        self.add(Dropout(rate=0.5))
        self.add(Dense(2, activation="softmax"))

