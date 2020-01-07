import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 2

img = cv.imread('C:/Users/Windows/Desktop/Trabalhos/Verso.png')  # trainImage
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
    marcat = img[y-10:y+10, x-1:x+19] # queryImage
for f in corners2:
    x,y = f.ravel()
    marcat2 = img[y-10:y+10, 519+x:539+x] # queryImage
for g in corners3:
    x,y = g.ravel()
    marcat3 = img[769+y:789+y, x-10:x+10] # queryImage
for h in corners4:
    x,y = h.ravel()
    marcat4 = img[769+y:789+y, 530+x:550+x] # queryImage

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(marcat,None)
kp2, des2 = sift.detectAndCompute(marca1,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
        
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = marcat.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img = cv.polylines(img,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv.drawMatches(marcat,kp1,img,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()
