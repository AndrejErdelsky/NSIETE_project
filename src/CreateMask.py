import os
import time
import glob
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import math
import numpy.matlib as npm


def convert5Pointto8Point(cx_, cy_, w_, h_, a_):
    """
    https://stackoverflow.com/questions/50204604/how-to-draw-a-filled-rotated-rectangle-with-center-coordinates-width-height-an
    funkcia ktora zo stredu, vysky, sirky a uhla otocenia bounding boxu vrati suradnice rotovaneho bounding boxu
    """
    theta = math.radians(a_)
    bbox = npm.repmat([[cx_], [cy_]], 1, 5) + \
           np.matmul([[math.cos(theta), math.sin(theta)],
                      [-math.sin(theta), math.cos(theta)]],
                     [[-w_ / 2, w_ / 2, w_ / 2, -w_ / 2, w_ / 2 + 8],
                      [-h_ / 2, -h_ / 2, h_ / 2, h_ / 2, 0]])

    x1, y1 = bbox[0][0], bbox[1][0]
    x2, y2 = bbox[0][1], bbox[1][1]
    x3, y3 = bbox[0][2], bbox[1][2]
    x4, y4 = bbox[0][3], bbox[1][3]

    return [x1, y1, x2, y2, x3, y3, x4, y4]


zaciatok = time.time()
directory = r'C:\Dataset\MaskCloudyRainySunnyUFPR04'
directorySave = r'C:\Dataset\FrameCloudyRainySunnyUFPR04'

color = (0, 0, 0)
thickness = -1

# parsovanie xmlka a ulozenie dvojic obrazok maska
filenames = glob.glob("C:\Dataset\PKLot\PKLot\PKLot\\UFPR04\*\\*\*.xml")
filenamesJPG = glob.glob("C:\Dataset\PKLot\PKLot\PKLot\\UFPR04\*\\*\*.jpg")
cisloobrazka = 0
cislo = 0
for filename in filenames:
    imageSave = Image.open(filenamesJPG[cisloobrazka])
    image = Image.new('L', (1280, 720), (0))
    tree = ET.parse(filename)
    root = tree.getroot()

    color_input = 0
    for movie in root.findall("./space[@occupied='0']"):
        center = movie.findall('rotatedRect/center')
        size = movie.findall('rotatedRect/size')
        angle = movie.findall('rotatedRect/angle')

        polygon = convert5Pointto8Point(int(center[0].attrib["x"]), int(center[0].attrib["y"]),
                                        int(size[0].attrib["w"]),
                                        int(size[0].attrib["h"]), ((-1) * (int(angle[0].attrib["d"]))))
        ImageDraw.Draw(image).polygon(polygon, outline='white', fill='white')

    os.chdir(directory)
    filename = "savedImage" + str(cislo) + ".jpg"
    cislo += 1

    # ulozenie masky
    image.save(filename)
    # ulozenie povodneho obrazka
    os.chdir(directorySave)
    imageSave.save(filename)
    cisloobrazka += 1
print(time.time() - zaciatok)
