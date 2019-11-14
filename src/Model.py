import keras.models
from keras.layers import *


class Baseline(keras.models.Sequential):

    def __init__(self):
        super(Baseline, self).__init__()
        self.model_layers = [
            InputLayer(input_shape=[64, 64, 1]),
            Conv2D(filters=32, kernel_size=5, strides=1, padding="same", activation="relu"),
            MaxPool2D(pool_size=5, padding="same"),

            Conv2D(filters=32, kernel_size=5, strides=1, padding="same", activation="relu"),
            MaxPool2D(pool_size=5, padding="same"),

            Conv2D(filters=32, kernel_size=5, strides=1, padding="same", activation="relu"),
            MaxPool2D(pool_size=5, padding="same"),
            Dropout(0.25),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(rate=0.5),
            Dense(2, activation="softmax")
        ]

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
