import os
import cv2
import glob
import xml.etree.ElementTree as ET


directory = r'C:\Dataset'
color = (0, 0, 0)
thickness = -1

# parse an xml file by name
filenames = glob.glob("C:\Dataset\PKLot\PKLot\PKLot\PUCPR\Sunny\\2012-09-11\*.xml")
filenamesJPG = glob.glob("C:\Dataset\PKLot\PKLot\PKLot\PUCPR\Sunny\\2012-09-11\*.jpg")
cisloobrazka=0
cislo=0
for filename in filenames:
    image = cv2.imread(filenamesJPG[cisloobrazka], 0)

    tree = ET.parse(filename)
    root = tree.getroot()

    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    for movie in root.findall("./space[@occupied='1']"):
        print(movie.attrib)
        i=0

        for point in movie.findall("./contour/point"):
            print(point.attrib["x"], point.attrib["y"])
            separator = ''
            startX = separator.join(point.attrib["x"])
            startY= separator.join(point.attrib["y"])
            if i==0:

                start_point = (int(startX), int(startY))
            if i==2:
                end_point = (int(startX), int(startY))
            i+=1

        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    os.chdir(directory)
    filename = "savedImage"+str(cislo)+".jpg"
    cislo+=1

    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename, image)
    cisloobrazka+=1
