from __future__ import absolute_import, division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import *
import keras
import matplotlib.pyplot as plt
from tensorflow_core.python.ops.gen_logging_ops import timestamp
import datetime
from src.LoadData import *
from src.Model import Baseline

tr_img_data, tr_lbl_data = ReshapeImages(TrainDataWithLabel(), -1, 64, 64, 1)
tst_img_data, tst_lbl_data = ReshapeImages(TestDataWithLabel(), -1, 64, 64, 1)
testing_images = TestDataWithLabel()
val_img_data = tr_img_data[len(tr_img_data)//2:]
val_lbl_data = tr_lbl_data[len(tr_lbl_data)//2:]
tr_img_data = tr_img_data[:len(tr_img_data)//2]
tr_lbl_data = tr_lbl_data[:len(tr_lbl_data)//2]
## Testovaci model na nacitanie datasetu
model = Baseline()

optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1,
        profile_batch=0)
]
model.fit(
    x=tr_img_data,
    y=tr_lbl_data,
    epochs=1,
    batch_size=100,
    callbacks=callbacks,
    validation_data=(val_img_data, val_lbl_data))

model.save("Baseline.h5")


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
