import cv2
import numpy as np

img = cv2.imread("database_files/exams/NM7510_P3_C_2019_2/NM7510_P3_pacote_C_2_sem_2019-44.png",1) 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('database_files/track_markers/template1.png',cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
loc = np.where(result >= 0.659)

for rt in zip(*loc[::-1]):
    cv2.rectangle(img, rt, (rt[0] + w, rt[1] + h), (0, 255, 0), 3)
    if rt[0] < 300 and rt[1]<200:
        m1x = rt[0]
        m1y = rt[1]
    if rt[0] > 300 and rt[0] < 1600 and rt[1]<200:
        m2x = rt[0]
        m2y = rt[1]
    if rt[0] < 300 and rt[1] > 2000:
        m3x = rt[0]
        m3y = rt[1]
    if rt[0] > 300 and rt[0] < 1600 and rt[1]>2000:
        m4x = rt[0]
        m4y = rt[1]
    if rt[0] > 1600 and rt[0] < 3000 and rt[1]<200:
        m5x = rt[0]
        m5y = rt[1]
    if rt[0] > 3000 and rt[1]<200:
        m6x = rt[0]
        m6y = rt[1]
    if rt[0] > 1600 and rt[0] < 3000 and rt[1]>2000:
        m7x = rt[0]
        m7y = rt[1]
    if rt[0] > 3000 and rt[1]>2000:
        m8x = rt[0]
        m8y = rt[1]

template2 = cv2.imread('database_files/track_markers/template2.png',cv2.IMREAD_GRAYSCALE)
r, f = template2.shape[::-1]

result2 = cv2.matchTemplate(gray_img, template2, cv2.TM_CCOEFF_NORMED)
loc2 = np.where(result2 >= 0.659)

for pt in zip(*loc2[::-1]):
    cv2.rectangle(img, pt, (pt[0] + r, pt[1] + f), (0, 255, 0), 3)
    if pt[0] < 300 and pt[1]<200:
        m1x = pt[0]
        m1y = pt[1]
    if pt[0] > 300 and pt[0] < 1600 and pt[1]<200:
        m2x = pt[0]
        m2y = pt[1]
    if pt[0] < 300 and pt[1] > 2000:
        m3x = pt[0]
        m3y = pt[1]
    if pt[0] > 300 and pt[0] < 1600 and pt[1]>2000:
        m4x = pt[0]
        m4y = pt[1]
    if pt[0] > 1600 and pt[0] < 3000 and pt[1]<200:
        m5x = pt[0]
        m5y = pt[1]
    if pt[0] > 3000 and pt[1]<200:
        m6x = pt[0]
        m6y = pt[1]
    if pt[0] > 1600 and pt[0] < 3000 and pt[1]>2000:
        m7x = pt[0]
        m7y = pt[1]
    if pt[0] > 3000 and pt[1]>2000:
        m8x = pt[0]
        m8y = pt[1]

print("m1: [{0},{1}]\nm2: [{2},{3}]\nm3: [{4},{5}]\nm4: [{6},{7}]".format(m1x, m1y, m2x, m2y, m3x, m3y, m4x, m4y))
print("m5: [{0},{1}]\nm6: [{2},{3}]\nm7: [{4},{5}]\nm8: [{6},{7}]".format(m5x, m5y, m6x, m6y, m7x, m7y, m8x, m8y))

lx1 = [m1x,m2x,m3x,m4x]
ly1 = [m1y,m2y,m3y,m4y]
lx2 = [m5x,m6x,m7x,m8x]
ly2 = [m5y,m6y,m7y,m8y]

if m1y>m2y:
    dy = m1y-m2y
    dx = m2x - m1x
    arc = np.degrees(np.arctan(dy/dx))
    print(arc)

    rows, cols, j = img.shape
    M = cv2.getRotationMatrix2D((cols//2,rows//2),-arc,1)
    imgf = cv2.warpAffine(img,M,(cols,rows))
elif m2y>m1y:
    dy = m2y-m1y
    dx = m2x-m1x    
    arc = np.degrees(np.arctan(dy/dx))
    print(arc)

    rows, cols, j = img.shape
    M = cv2.getRotationMatrix2D((cols//2,rows//2),arc,1)
    imgf = cv2.warpAffine(img,M,(cols,rows))
elif m1y == m2y:
    imgf = img

if m7y>m8y:
    dy2 = m7y-m8y
    dx2 = m8x - m7x
    arc2 = np.degrees(np.arctan(dy2/dx2))
    print(arc2)

    rows, cols, j = img.shape
    M = cv2.getRotationMatrix2D((cols//2,rows//2),-arc2,1)
    imgf2 = cv2.warpAffine(img,M,(cols,rows))
elif m8y>m7y:
    dy2 = m8y-m7y
    dx2 = m8x-m7x    
    arc2 = np.degrees(np.arctan(dy2/dx2))
    print(arc2)

    rows, cols, j = img.shape
    M = cv2.getRotationMatrix2D((cols//2,rows//2),arc2,1)
    imgf2 = cv2.warpAffine(img,M,(cols,rows))
elif m8y == m7y:
    imgf2 = img

img1 = imgf[ min(ly1, key=int)+int(h/2): max(ly1, key=int),  min(lx1, key=int)+int(w/1.5): max(lx1, key=int)+int(w/3.5)]
img2 = imgf2[ min(ly2, key=int)+int(h/2): max(ly2, key=int)+int(h/2),  min(lx2, key=int)+int(w/2): max(lx2, key=int)+int(w/2)]

img1 = img1[::3,::3]
img2 = img2[::3,::3]
cv2.imshow("img", img1)
cv2.imshow("img2", img2)
cv2.waitKey(0)