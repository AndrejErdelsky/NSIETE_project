from keras.models import Sequential, Model
from keras.layers import *


class Baseline(Sequential):
    def __init__(self):
        super(Baseline, self).__init__(name='Baseline')
        # self.add(InputLayer(input_shape=[64, 64, 1]))
        self.add(Conv2D(filters=16, kernel_size=11, strides=4, padding="same", activation="relu", input_shape=(64, 64, 1)))
        self.add(MaxPool2D(pool_size=3, strides=2, padding="same"))

        self.add(Conv2D(filters=20, kernel_size=5, strides=1, padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=3, strides=2, padding="same"))

        self.add(Conv2D(filters=30, kernel_size=3, strides=1, padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=3, strides=2, padding="same"))

        self.add(Flatten())
        self.add(Dense(48, activation="relu"))
        self.add(Dense(2, activation="softmax"))


def get_segmentation_model():
    inp = Input(shape=(1280, 640, 3,))

    c1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", strides=2)(inp)
    c2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", strides=2)(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", strides=2)(p1)
    c4 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", strides=2)(c3)
    p2 = MaxPooling2D(pool_size=(2, 2))(c4)

    up1 = convolutional.UpSampling2D(size=(2, 2))(p2)
    dc1 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(
        up1)
    dc2 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(
        dc1)

    up2 = convolutional.UpSampling2D(size=(2, 2))(dc2)
    dc3 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(
        up2)
    dc4 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(
        dc3)

    out = Conv2D(kernel_size=(1, 1), filters=1, padding='same', activation='sigmoid')(dc4)

    model = Model(input=inp, output=out)
    return model
