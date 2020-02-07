import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
# //wczytywanie zdjęcia


myfile = open('rece.txt', 'w')

for filename in os.listdir("./rockpaperscissors/paper/"):
    image = cv2.imread("./rockpaperscissors/paper/" + filename)
    # //odcienie szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # //detekcja cech
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=600)
    (kps, descs) = surf.detectAndCompute(gray, None)
    for i, j in zip(kps, descs):
        myfile.writelines("A,"+ np.array2string(j, separator=',').replace('\n', '') + '\n')


for filename in os.listdir("./rockpaperscissors/rock/"):
    image = cv2.imread("./rockpaperscissors/rock/" + filename)
    # //odcienie szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # //detekcja cech
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=600)
    (kps, descs) = surf.detectAndCompute(gray, None)
    for i, j in zip(kps, descs):
        myfile.writelines("Q," + np.array2string(j, separator=',').replace('\n', '') + '\n')
    # print(len(kps))
    # img2 = cv2.drawKeypoints(image,kps,None,(255,0,0),4)

    # plt.imshow(img2),plt.show()

    # raise Exception

myfile.close()