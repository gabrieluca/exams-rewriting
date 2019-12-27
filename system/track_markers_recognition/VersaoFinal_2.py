import cv2
import numpy as np

img = cv2.imread("database_files/exams/NM7510_P3_2019_2/NM7510_P3_pacote_A_2_sem_2019-01.png",1) 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('database_files/track_markers/template3.png',cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
loc = np.where(result >= 0.659)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

template2 = cv2.imread('database_files/track_markers/template4.png',cv2.IMREAD_GRAYSCALE)
r, f = template2.shape[::-1]

result2 = cv2.matchTemplate(gray_img, template2, cv2.TM_CCOEFF_NORMED)
loc2 = np.where(result2 >= 0.659)

for pt in zip(*loc2[::-1]):
    cv2.rectangle(img, pt, (pt[0] + r, pt[1] + f), (0, 255, 0), 3)

img = cv2.resize(img, (1080, 660))
cv2.imshow("img", img)
cv2.waitKey(0)

