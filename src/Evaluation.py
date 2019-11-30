from PIL import Image
from keras.models import load_model
from tensorboard.notebook import display

from src.LoadData import TestDataWithLabel
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from src.LoadDataSegmentation import *
import cv2

model = load_model('ModelDatasetUFPR04.h5')
# plot_model(model, to_file='model_vizualization.png')
model.summary()
fig = plt.figure(figsize=(14, 14))
y = fig.add_subplot(6, 5, 1)
testing_images = TestDataWithLabel()

for cnt, data in enumerate(testing_images[160:179]):
    y = fig.add_subplot(6, 5, cnt + 1)
    img = data
    # data = ReshapeImages(img, 1, 1280, 720, 3)
    print(data.shape)
    model_out = model.predict(np.array(data).reshape(1, 320, 320, 3))
    # model_out = np.argmax(model_out, axis=-1)
    print(np.max(model_out))
    #model_out = np.round(model_out, 2)

    model_out *= 255
    model_out = np.reshape(model_out, (320, 320))
    model_out = model_out.astype(np.uint8)
    # print(type(model_out))
    # print(model_out.shape)

    print(model_out[0])

    # model_out = model_out.reshape(174,314)
    img = Image.fromarray(model_out, 'L')
    # img.save('my.png')
    # img = img.resize((1280, 720))
    # img = img.resize((1280, 720), resample=Image.BILINEAR)
    #img.show()
    # model_out.save("ahoj.jpg")
    # print(type(model_out),model_out)
    # cv2.imshow('test',model_out)
    # cv2.waitKey(0)
    # break
    y.imshow(img)

    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
#   return pred_mask[0]
#
# def show_predictions(dataset=None, num=1):
#   if dataset:
#     for image, mask in dataset.take(num):
#       pred_mask = model.predict(image)
#       display([image[0], mask[0], create_mask(pred_mask)])
#   else:
#     display([sample_image, sample_mask,
#              create_mask(model.predict(sample_image[tf.newaxis, ...]))])
# plt.interactive(False)
# fig = plt.figure(figsize=(14, 14))
# for cnt, data in enumerate(testing_images[0:18]):
#
#     y = fig.add_subplot(6, 5, cnt + 1)
#     img = data[0]
#     data = img.reshape(1, 1280, 720, 3)
#     model_out = model.predict([data])
#
#     y.imshow(img, cmap="gray")
#
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_yaxis().set_visible(False)
# plt.show()
