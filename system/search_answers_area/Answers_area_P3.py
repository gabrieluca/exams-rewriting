import cv2
import numpy as np

img = cv2.imread("database_files/exams/NM7510_P3_C_2019_2/NM7510_P3_pacote_C_2_sem_2019-01.png",1) 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('database_files/track_markers/template3.png',cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
loc = np.where(result >= 0.659)

for rt in zip(*loc[::-1]):
    cv2.rectangle(img, rt, (rt[0] + w, rt[1] + h), (0, 255, 0), 3)

template2 = cv2.imread('database_files/track_markers/template4.png',cv2.IMREAD_GRAYSCALE)
r, f = template2.shape[::-1]

result2 = cv2.matchTemplate(gray_img, template2, cv2.TM_CCOEFF_NORMED)
loc2 = np.where(result2 >= 0.659)

for pt in zip(*loc2[::-1]):
    cv2.rectangle(img, pt, (pt[0] + r, pt[1] + f), (0, 255, 0), 3)

if pt[0]>1600:
    img1 = img[int(pt[0]/50)+int(w*1.8):rt[1] + int(f/2.5), pt[0]+int(w*0.7):rt[0]+int(w*0.5)]
    img2 = img[int(pt[0]/50)+w+1628:rt[1] + int(f/2.95), pt[0]-1625:rt[0]+w-1685]
    img1 = cv2.resize(img1, (1080, 660))
    img2 = cv2.resize(img2, (1080, 660))
    cv2.imshow("img", img1)
    cv2.imshow("img2", img2)
    print('{0}, {1}, {2}, {3}'.format(pt[0],pt[1],rt[0],rt[1]))
    cv2.waitKey(0)
else: 
    if rt[0]<1650:
        img1 = img[int(pt[0]/50)+w+1519:rt[1] + int(f*0.5), pt[0]+int(w*0.4):rt[0]+int(w/2.5)]
        img2 = img[int(pt[0]/50)+int(w*1.95):rt[1] + int(f*0.5), pt[0]+1680:rt[0]+w+1619]
        img1 = cv2.resize(img1, (1080, 660))
        img2 = cv2.resize(img2, (1080, 660))
        cv2.imshow("img", img1)
        cv2.imshow("img2", img2)
        print('{0}, {1}, {2}, {3}'.format(pt[0],pt[1],rt[0],rt[1]))
        cv2.waitKey(0)
    else:
        img1 = img[int(pt[0]/50)+w+1510:rt[1] + int(f*0.45), pt[0]+int(w*0.5):rt[0]+int(w/1.5)-1658]
        img2 = img[int(pt[0]/50)+int(w*2.2):rt[1] + int(f*0.5), pt[0]+1687:rt[0]+int(w*0.5)]
        img1 = cv2.resize(img1, (1080, 660))
        img2 = cv2.resize(img2, (1080, 660))
        cv2.imshow("img", img1)
        cv2.imshow("img2", img2)
        print('{0}, {1}, {2}, {3}'.format(pt[0],pt[1],rt[0],rt[1]))
        cv2.waitKey(0)
