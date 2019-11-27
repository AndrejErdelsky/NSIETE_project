from __future__ import absolute_import, division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import *
import keras
import matplotlib.pyplot as plt
# from tensorflow_core.python.ops.gen_logging_ops import timestamp
import datetime
from src.LoadDataSegmentation import *
from src.Model import *
#tst_img_data, tst_lbl_data = ReshapeImages(TrainDataWithLabel(), -1, 64, 64, 3)
final_data = ReshapeImages(TrainDataWithLabel(),-1, 1280, 720, 3)
mask =  ReshapeImages(LoadMask(), -1, 1280, 720, 1)
val_data = final_data[len(final_data)//2:]
tr_data = final_data[:len(final_data)//2]
val_mask = mask[len(mask)//2:]
tr_mask = mask[:len(mask)//2]
## Testovaci model na nacitanie datasetu
model = get_segmentation_model2()

optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1,
        profile_batch=0
    )
]
model.fit(
    x=tr_data,
    y=tr_mask,
    epochs=1,
    batch_size=1,
    callbacks=callbacks,
    validation_data=(val_data, val_mask))

model.save("Baseline.h5")


