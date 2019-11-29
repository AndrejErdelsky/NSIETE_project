from __future__ import absolute_import, division, print_function, unicode_literals

from keras import metrics
from keras.models import Sequential
from keras.layers import *
import keras
import matplotlib.pyplot as plt
# from tensorflow_core.python.ops.gen_logging_ops import timestamp
import datetime
from src.LoadDataSegmentation import *
from src.Model import *
from sklearn.utils.class_weight import compute_class_weight


# tst_img_data, tst_lbl_data = ReshapeImages(TrainDataWithLabel(), -1, 64, 64, 3)
final_data = TrainDataWithLabel()
mask = LoadMask()
val_data = final_data[12:]
tr_data = final_data[:1]
val_mask = mask[12:]
tr_mask = mask[:1]
## Testovaci model na nacitanie datasetu
input_img = Input(shape=(320,320,3))
model = get_unet(input_img)

optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['accuracy'],
              #sample_weight_mode='temporal'
              )

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1#,
        #profile_batch=0
    )

]

#tr_mask = np.ravel(np.round(tr_mask).astype(int))
# print(np.unique(np.expand_dims(np.ravel(tr_mask),axis=0)))
# print(model.summary())
# cw = compute_class_weight('balanced', classes=np.unique(tr_mask), y=tr_mask)
print(np.sum(tr_mask)/np.array(tr_mask).size)
model.fit(
    x=np.array(tr_data),
    y=np.expand_dims(tr_mask, axis=-1),
    #sample_weight=np.expand_dims(np.array([cw[i] for i in tr_mask]),axis=0),
    epochs=60,
    batch_size=3#,
    #callbacks=callbacks
    #validation_data=(np.array(val_data), np.expand_dims(val_mask,axis=-1))
)

model.save("Baseline2.h5")
