from keras.models import load_model
from src.LoadData import TestDataWithLabel
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model

model = load_model('Baseline.h5',compile=True)
# plot_model(model, to_file='model_vizualization.png')
model.summary()

testing_images = TestDataWithLabel()

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
