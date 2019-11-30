from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import os
from random import shuffle
from tqdm import tqdm
from keras.optimizers import *
import numpy as np


#nastavenie ciest k datasetu
#Sunny05
data05S = "..\Dataset\Snimky"
mask05S = "..\Dataset\Mask"

#CloudyRainy05
data05CR = "..\DatasetCloudyRainy\Snimky"
mask05CR = "..\DatasetCloudyRainyt\Mask"

#PUCR
dataPUCR = "..\DatasetPUCR\Snimky"
maskPUCR = "..\DatasetPUCR\Mask"

#04
data04 = "..\DatasetUFPR04\Snimky"
mask04 = "..\DatasetUFPR04\Mask"

data=data04
mask=mask04

#olablovanie datasetu, trenovacie data
def TrainDataWithLabel():
    images = []
    #for picture in data:
    for i in tqdm(os.listdir(data)):
            path = os.path.join(data,i)
            img = cv2.imread(path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(320,320))
            img = img / 255
            images.append(np.array(img))


    return images

def TestDataWithLabel():
    images = []
    # for picture in data:
    for i in tqdm(os.listdir(data)):
        path = os.path.join(data, i)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 320))
        img = img / 255
        images.append(np.array(img))

    return images

def LoadMask():
    images = []
    #for picture in mask:
    for i in tqdm(os.listdir(mask)):
            path = os.path.join(mask,i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img,(320,320))
            cv2.imwrite("test.jpg", img)
            img = img / 255
            images.append(np.array(img))

    return images



def ReshapeImages(images, batch, width, height):
    img_data = []
    for i in images:
        img_data.append(cv2.resize(i,(width,height)))



    return img_data