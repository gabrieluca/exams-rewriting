import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('C:/Users/Windows/Desktop/Trabalhos/Verso.png')
marca1 = img[0:57, 0:57]
marca2 = img[0:57, 540:590]
marca3 = img[790:840, 0:57]
marca4 = img[790:840, 540:590]
gray1 = cv.cvtColor(marca1,cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(marca2,cv.COLOR_BGR2GRAY)
gray3 = cv.cvtColor(marca3,cv.COLOR_BGR2GRAY)
gray4 = cv.cvtColor(marca4,cv.COLOR_BGR2GRAY)

corners1 = cv.goodFeaturesToTrack(gray1, 1,0.9,10)
corners2 = cv.goodFeaturesToTrack(gray2, 1,0.9,10)
corners3 = cv.goodFeaturesToTrack(gray3, 1,0.9,10)
corners4 = cv.goodFeaturesToTrack(gray4, 1,0.9,10)
corners1 = np.int0(corners1)
corners2 = np.int0(corners2)
corners3 = np.int0(corners3)
corners4 = np.int0(corners4)
for i in corners1:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)
for f in corners2:
    x,y = f.ravel()
    cv.circle(img,(540+x,y),3,255,-1)
for g in corners3:
    x,y = g.ravel()
    cv.circle(img,(x,790+y),3,255,-1)
for h in corners4:
    x,y = h.ravel()
    cv.circle(img,(540+x,790+y),3,255,-1)

plt.imshow(img),plt.show()
