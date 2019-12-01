from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
from src.LoadDataSegmentation import *

f = open("../evaluacia.txt", "a")

#NazvyModelov=["ModelCloudyRainyUFPR05.h5","ModelCloudyRainyUFPR05.h5","UFPR05CRE15.h5","UFPR05CRE20.h5","UFPR05CRLossBCE.h5","UFPR05CRLossBCE.h5"]
#NazvyModelov=["PUCR.h5","PUCRE15.h5","PUCRE18.h5","PUCRE5.h5","PUCRLossBCE.h5"]
#NazvyModelov=["UFPR04.h5","UFPR04E10.h5"," UFPR04E15.h5","UFPR04E5.h5","UFPR04LossBCE.h5"]
NazvyModelov=["UFPR05S.h5","UFPR05SE15.h5","UFPR05SE5.h5","UFPR05SLossBCE.h5"]

testing_images = TestDataWithLabel()
masks = LoadMask()
for nazov in NazvyModelov:
    model = load_model(nazov)
    model.summary()
    # fig = plt.figure(figsize=(14, 14))
    # y = fig.add_subplot(6, 5, 1)

    differences=[]

    for cnt, data in enumerate(testing_images[0:1000]):
        # y = fig.add_subplot(6, 5, cnt + 1)
        img = data
        maska = masks[cnt]
        # data = ReshapeImages(img, 1, 1280, 720, 3)
        #print(data.shape)
        model_out = model.predict(np.array(data).reshape(1, 320, 320, 3))
        # model_out = np.argmax(model_out, axis=-1)
        model_out *= 255
        model_out = np.reshape(model_out, (320, 320))
        #print(type(maska), maska.shape)
        #print(type(model_out), model_out.shape)
        intersection = np.logical_and(np.array(maska), np.array(model_out))
        union = np.logical_or(np.array(maska), np.array(model_out))
        iou_score = np.sum(intersection) / np.sum(union)
        differences.append(iou_score)

        # print(np.max(model_out))
        # model_out = np.round(model_out, 2)
        #
        # model_out *= 255
        # model_out = np.reshape(model_out, (320, 320))
        # model_out = model_out.astype(np.uint8)
        #
        # print(model_out[0])
        #
        # img = Image.fromarray(model_out, 'L')

        #img.show()
    print(differences)
    print(np.sum(np.array(differences))/len(differences))
    uspesnost = np.sum(np.array(differences))/len(differences)
    f.write("Uspesnost modelu "+str(nazov)+" je " + str(uspesnost))
f.close()

#     y.imshow(img)
#
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_yaxis().set_visible(False)
# plt.show()
