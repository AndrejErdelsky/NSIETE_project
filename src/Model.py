from keras.models import Sequential, Model
from keras.layers import *


class Baseline(Sequential):
    def __init__(self):
        super(Baseline, self).__init__(name='Baseline')
        # self.add(InputLayer(input_shape=[64, 64, 1]))
        self.add(
            Conv2D(filters=16, kernel_size=11, strides=4, padding="same", activation="relu", input_shape=(64, 64, 1)))
        self.add(MaxPool2D(pool_size=3, strides=2, padding="same"))

        self.add(Conv2D(filters=20, kernel_size=5, strides=1, padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=3, strides=2, padding="same"))

        self.add(Conv2D(filters=30, kernel_size=3, strides=1, padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=3, strides=2, padding="same"))

        self.add(Flatten())
        self.add(Dense(48, activation="relu"))
        self.add(Dense(2, activation="softmax"))


def get_segmentation_model():
    inp = Input(shape=(640, 640, 3,))

    c1 = Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu", strides=2)(inp)
    c2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", strides=1)(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(filters=128, kernel_size=(5, 5), padding="same", activation="relu", strides=1)(p1)
    c4 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", strides=1)(c3)
    p2 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", strides=1)(p2)
    c6 = Conv2D(kernel_size=(1, 1), filters=128, padding='same', activation='relu')(c5)

    up1 = convolutional.UpSampling2D(size=(2, 2), interpolation='bilinear')(c6)
    dc1 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=1)(
         up1)
    dc2 = Deconvolution2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', strides=1)(
         dc1)
    # dc2 = Deconvolution2D(filters=32, kernel_size=(7, 7), padding='valid', activation='relu', strides=2)(
    #     dc2)
    #
    # up2 = convolutional.UpSampling2D(size=(2, 2))(dc2)
    # dc3 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(
    #     up2)
    # dc4 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(
    #     dc3)

    out = Conv2D(kernel_size=(1, 1), filters=16, padding='same', activation='relu')(dc2)
    out = Conv2D(kernel_size=(1, 1), filters=8, padding='same', activation='relu')(out)
    out = Conv2D(kernel_size=(1, 1), filters=2, padding='same', activation='softmax')(out)

    out = Reshape((25600, 2))(out)
    model = Model(input=inp, output=out)
    return model


def get_segmentation_model2():
    inp = Input(shape=( 720,1280, 3,))

    c1 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(inp)
    c2 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(p1)
    c4 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(c3)
    p2 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(p2)
    c6 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(c5)
    p3 = MaxPooling2D(pool_size=(2, 2))(c6)

    up1 = convolutional.UpSampling2D(size=(2, 2))(p3)
    dc1 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        up1)
    dc2 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        dc1)

    up2 = convolutional.UpSampling2D(size=(2, 2))(dc2)
    dc3 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        up2)
    dc4 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        dc3)

    up3 = convolutional.UpSampling2D(size=(2, 2))(dc4)
    dc5 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        up3)
    dc6 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        dc5)
    dc7 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        dc6)
    dc8 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        dc7)

    out = Conv2D(kernel_size=(1, 1), filters=1, padding='same', activation='sigmoid')(dc8)

    model = Model(input=inp, output=out)
    return model

def get_segmentation_model3():
    inp = Input(shape=(720, 1280,  3,))

    c1 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(inp)
    c2 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(p1)
    c4 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(c3)
    p2 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(p2)
    c6 = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(c5)
    p3 = MaxPooling2D(pool_size=(2, 2))(c6)

    up1 = convolutional.UpSampling2D(size=(2, 2))(p3)
    dc1 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        up1)
    dc2 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        dc1)

    up2 = convolutional.UpSampling2D(size=(2, 2))(dc2)
    dc3 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        up2)
    dc4 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        dc3)

    up3 = convolutional.UpSampling2D(size=(2, 2))(dc4)
    dc5 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        up3)
    dc6 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        dc5)
    dc7 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        dc6)
    dc8 = Deconvolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=1)(
        dc7)

    out = Conv2D(kernel_size=(1, 1), filters=1, padding='same', activation='sigmoid')(dc8)

    model = Model(input=inp, output=out)
    return model


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model