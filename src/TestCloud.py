import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import *
import datetime
import os


# model
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


# nacitanie modelu
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1  # ,
        # profile_batch=0
    )

]

# definovanie generatora
train_datagen = ImageDataGenerator()
mask_train_datagen = ImageDataGenerator(rescale=1 / 255)
val_datagen = ImageDataGenerator()

train_image_generator = train_datagen.flow_from_directory(
    '/NN/xerdelsky/NSIETE_project/Dataset/train_frames',
    target_size=(320, 320),
    class_mode=None,
    batch_size=1)

train_mask_generator = mask_train_datagen.flow_from_directory(
    '/NN/xerdelsky/NSIETE_project/Dataset/train_masks',
    target_size=(320, 320),
    class_mode=None,
    color_mode='grayscale',
    batch_size=1)

val_image_generator = val_datagen.flow_from_directory(
    '/NN/xerdelsky/NSIETE_project/Dataset/val_frames',
    target_size=(320, 320),
    class_mode=None,
    batch_size=1)

val_mask_generator = val_datagen.flow_from_directory(
    '/NN/xerdelsky/NSIETE_project/Dataset/val_masks',
    target_size=(320, 320),
    class_mode=None,
    color_mode='grayscale',
    batch_size=1)

NO_OF_VAL_IMAGES = len(os.listdir('/NN/xerdelsky/NSIETE_project/Dataset/val_frames/val'))
NO_OF_TRAIN_IMAGES = len(os.listdir('/NN/xerdelsky/NSIETE_project/Dataset/train_frames/train'))
BATCH_SIZE = 'Batch size previously initialised'
train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

input_img = keras.layers.Input(shape=(320, 320, 3))
model = get_unet(input_img)
model.summary()

optimizer = keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['accuracy'],
              # sample_weight_mode='temporal'
              )
model.fit_generator(generator=train_generator, epochs=30, steps_per_epoch=1, callbacks=callbacks)

model.save('ModelDatasetUFPR05frame1.h5')
