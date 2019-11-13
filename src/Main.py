from __future__ import absolute_import, division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt
from src.LoadData import *

tr_img_data, tr_lbl_data = ReshapeImages(TrainDataWithLabel(), -1, 64, 64, 1)
tst_img_data, tst_lbl_data = ReshapeImages(TestDataWithLabel(), -1, 64, 64, 1)
testing_images = TestDataWithLabel()


## Testovaci model na nacitanie datasetu
model = Sequential()
model.add(InputLayer(input_shape=[64, 64, 1]))
model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=5, padding="same"))


model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation="relu"))

optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(tr_img_data, tr_lbl_data, epochs=1, batch_size=100)

model.summary()
plt.interactive(False)
fig = plt.figure(figsize=(14, 14))
for cnt, data in enumerate(testing_images[10:40]):

    y = fig.add_subplot(6, 5, cnt + 1)
    img = data[0]
    data = img.reshape(1, 64, 64, 1)
    model_out = model.predict([data])
    if np.argmax(model_out) == 1:
        str_label = "occupied"
    else:
        str_label = "empty"
    y.imshow(img, cmap="gray")
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()