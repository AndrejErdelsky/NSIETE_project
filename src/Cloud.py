import keras
from IPython.display import Image
import os
import re
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import *

DATA_PATH = '/NN/xerdelsky/NSIETE_project/Dataset/'
FRAME_PATH = DATA_PATH + 'Snimky/'
MASK_PATH = DATA_PATH + 'Mask/'

folders = ['train_frames/train', 'train_masks/train', 'val_frames/val', 'val_masks/val', 'test_frames/test',
           'test_masks/test']
for folder in folders:
    os.makedirs(DATA_PATH + folder)

all_frames = os.listdir(FRAME_PATH)
all_masks = os.listdir(MASK_PATH)

all_frames.sort(key=lambda var: [int(x) if x.isdigit() else x
                                 for x in re.findall(r'[^0-9]|[0-9]+', var)])
all_masks.sort(key=lambda var: [int(x) if x.isdigit() else x
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])

train_split = int(0.7 * len(all_frames))
val_split = int(0.9 * len(all_frames))

train_frames = all_frames[:train_split]
val_frames = all_frames[train_split:val_split]
test_frames = all_frames[val_split:]

train_masks = [f for f in all_masks if f in train_frames]
val_masks = [f for f in all_masks if f in val_frames]
test_masks = [f for f in all_masks if f in test_frames]


def add_frames(dir_name, image):
    img = Image.open(FRAME_PATH + image)
    img.save(DATA_PATH + '/{}'.format(dir_name) + '/' + image)


def add_masks(dir_name, image):
    img = Image.open(MASK_PATH + image)
    img.save(DATA_PATH + '/{}'.format(dir_name) + '/' + image)


frame_folders = [(train_frames, 'train_frames/train'), (val_frames, 'val_frames/val'),
                 (test_frames, 'test_frames/test')]

mask_folders = [(train_masks, 'train_masks/train'), (val_masks, 'val_masks/val'),
                (test_masks, 'test_masks/test')]

for folder in frame_folders:
    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_frames, name, array))

for folder in mask_folders:
    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_masks, name, array))

# definovanie generatora
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_image_generator = train_datagen.flow_from_directory(
    '/NN/xerdelsky/NSIETE_project/Dataset/train_frames',
    target_size=(320, 320),
    class_mode=None,
    batch_size=2)

train_mask_generator = train_datagen.flow_from_directory(
    '/NN/xerdelsky/NSIETE_project/Dataset/train_masks',
    target_size=(320, 320),
    class_mode=None,
    color_mode='grayscale',
    batch_size=2)

val_image_generator = val_datagen.flow_from_directory(
    '/NN/xerdelsky/NSIETE_project/Dataset/val_frames',
    target_size=(320, 320),
    class_mode=None,
    batch_size=2)

val_mask_generator = val_datagen.flow_from_directory(
    '/NN/xerdelsky/NSIETE_project/Dataset/val_masks',
    target_size=(320, 320),
    class_mode=None,
    color_mode='grayscale',
    batch_size=2)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)


# vrstva 2 konvolucnych sieti
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
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
    # Cesta zmensenia
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

    # cesta rekonstrukcie
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
input_img = keras.layers.Input(shape=(320, 320, 3))
model = get_unet(input_img)
model.summary()

optimizer = keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['accuracy'],
              # sample_weight_mode='temporal'
              )
model.fit_generator(generator=train_generator, epochs=2, steps_per_epoch=5)

model.save('Model.h5')
