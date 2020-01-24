import cv2
import numpy as np
from scipy.ndimage import rotate

def rotate_image(mat, angle):

    height, width, _ = mat.shape
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

img = cv2.imread("database_files/exams/NM7510_P3_2019_2/NM7510_P3_pacote_A_2_sem_2019-06.png",1) 
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

img = rotate_image(img,-0.5)

if pt[0]>1600:
    img1 = img[int(pt[0]/50)+w:rt[1] + f, pt[0]:rt[0]+w]
    img2 = img[int(pt[0]/50)+w+470:rt[1] + f, pt[0]-1650:rt[0]+w-1650]
    img1 = cv2.resize(img1, (1080, 660))
    img2 = cv2.resize(img2, (1080, 660))
    cv2.imshow("img", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
else:
    img1 = img[int(pt[0]/50)+w+340:rt[1] + f, pt[0]:rt[0]+w]
    img2 = img[int(pt[0]/50)+w:rt[1] + f, pt[0]+1650:rt[0]+w+1650]
    img1 = cv2.resize(img1, (1080, 660))
    img2 = cv2.resize(img2, (1080, 660))
    cv2.imshow("img", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
