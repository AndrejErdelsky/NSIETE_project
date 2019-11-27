from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import os
from random import shuffle
from tqdm import tqdm
from keras.optimizers import *
import numpy as np


#nastavenie ciest k datasetu
#Sunny
data = "C:\Dataset\Snimky"
mask = "C:\Dataset\Mask"



#olablovanie datasetu, trenovacie data
def TrainDataWithLabel():
    images = []
    #for picture in data:
    for i in tqdm(os.listdir(data)):
            path = os.path.join(data,i)
            img = cv2.imread(path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(1280,720))
            images.append([np.array(img)])

    return images

def LoadMask():
    images = []
    #for picture in mask:
    for i in tqdm(os.listdir(mask)):
            path = os.path.join(mask,i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(1280,720))
            images.append([np.array(img)])

    return images



def ReshapeImages(images, x, y, z, w):

    img_data = np.array([i for i in images]).reshape(x, y, z, w)


    return img_data