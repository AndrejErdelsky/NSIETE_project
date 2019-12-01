from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
from src.LoadDataSegmentation import *

model = load_model('ModelDatasetPUCR.h5')
model.summary()
fig = plt.figure(figsize=(14, 14))
y = fig.add_subplot(6, 5, 1)
testing_images = TestDataWithLabel()

for cnt, data in enumerate(testing_images[:29]):
    y = fig.add_subplot(6, 5, cnt + 1)
    img = data
    # data = ReshapeImages(img, 1, 1280, 720, 3)
    print(data.shape)
    model_out = model.predict(np.array(data).reshape(1, 320, 320, 3))
    # model_out = np.argmax(model_out, axis=-1)
    print(np.max(model_out))
    model_out = np.round(model_out, 2)

    model_out *= 255
    model_out = np.reshape(model_out, (320, 320))
    model_out = model_out.astype(np.uint8)

    print(model_out[0])

    img = Image.fromarray(model_out, 'L')
    img.show()
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
