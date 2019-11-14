from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import os
from random import shuffle
from tqdm import tqdm
from keras.optimizers import *
import numpy as np


#nastavenie ciest k datasetu
#Sunny
train_data_Empty_S = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Sunny\\2012-09-11\Empty"
train_data_Occupied_S = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Sunny\\2012-09-11\Occupied"
test_data_Empty_S = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Sunny\\2012-09-14\Empty"
test_data_Occupied_S = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Sunny\\2012-09-14\Occupied"
#Rainy
train_data_Empty_R = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Rainy\\2012-09-16\Empty"
train_data_Occupied_R = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Rainy\\2012-09-16\Occupied"
test_data_Empty_R = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Rainy\\2012-10-11\Empty"
test_data_Occupied_R = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Rainy\\2012-10-11\Occupied"
#Cloudy
train_data_Empty_C = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Cloudy\\2012-09-12\Empty"
train_data_Occupied_C = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Cloudy\\2012-09-12\Occupied"
test_data_Empty_C = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Cloudy\\2012-10-05\Empty"
test_data_Occupied_C = "C:\Dataset\PKLot\PKLot\PKLotSegmented\PUC\Cloudy\\2012-10-05\Occupied"

train_data_Empty1 = [train_data_Empty_C] + [train_data_Empty_R] + [test_data_Empty_R]
train_data_Occupied1 = [train_data_Occupied_C]+[test_data_Occupied_R]+ [test_data_Occupied_S]
test_data_Empty1 = [test_data_Empty_R] + [test_data_Empty_C] + [test_data_Empty_S]
test_data_Occupied1 = [test_data_Occupied_S] + [test_data_Occupied_R] + [test_data_Occupied_C]
#olablovanie datasetu, trenovacie data
def TrainDataWithLabel():
    train_images = []
    for train_data_Empty in train_data_Empty1:
        for i in tqdm(os.listdir(train_data_Empty)):
            path = os.path.join(train_data_Empty,i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(64,64))
            train_images.append([np.array(img),0])
    for train_data_Occupied in train_data_Occupied1:
        for i in tqdm(os.listdir(train_data_Occupied)):
            path = os.path.join(train_data_Occupied,i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(64,64))
            train_images.append([np.array(img),1])
    shuffle(train_images)
    return train_images

#olablovanie datasetu, testovacie data
def TestDataWithLabel():
    test_images = []
    for test_data_Empty in test_data_Empty1:
        for i in tqdm(os.listdir(test_data_Empty)):
            path = os.path.join(test_data_Empty, i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            test_images.append([np.array(img), 0])
    for test_data_Occupied in test_data_Occupied1:
        for i in tqdm(os.listdir(test_data_Occupied)):
            path = os.path.join(test_data_Occupied, i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            test_images.append([np.array(img), 1])
    shuffle(test_images)

    return test_images

def ReshapeImages(images, x, y, z, w):

    img_data = np.array([i[0] for i in images]).reshape(x, y, z, w)
    lbl_data = np.array([i[1] for i in images])

    return img_data, lbl_data