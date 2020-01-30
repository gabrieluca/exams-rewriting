import cv2
import numpy as np

def normalize_crop(img,template,template2):
    #turn image to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #save the templates shapes
    w, h = template.shape[::-1]
    r, f = template2.shape[::-1]
    # find the first template within the image
    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= 0.659)

    # draw the marks and save their respective values
    for rt in zip(*loc[::-1]):
        cv2.rectangle(img, rt, (rt[0] + w, rt[1] + h), (0, 255, 0), 3)
        if rt[0] < 300 and rt[1]<200:
            m1 = [rt[0],rt[1]]
        if rt[0] > 300 and rt[0] < 1600 and rt[1]<200:
            m2 = [rt[0],rt[1]]
        if rt[0] < 300 and rt[1] > 2000:
            m3 = [rt[0],rt[1]]
        if rt[0] > 300 and rt[0] < 1600 and rt[1]>2000:
            m4 = [rt[0],rt[1]]
        if rt[0] > 1600 and rt[0] < 3000 and rt[1]<200:
            m5 = [rt[0],rt[1]]
        if rt[0] > 3000 and rt[1]<200:
            m6 = [rt[0],rt[1]]
        if rt[0] > 1600 and rt[0] < 3000 and rt[1]>2000:
            m7 = [rt[0],rt[1]]
        if rt[0] > 3000 and rt[1]>2000:
            m8 = [rt[0],rt[1]]

    # find the second template within the image
    result2 = cv2.matchTemplate(gray_img, template2, cv2.TM_CCOEFF_NORMED)
    loc2 = np.where(result2 >= 0.659)

    # draw the marks and save their respective values
    for pt in zip(*loc2[::-1]):
        cv2.rectangle(img, pt, (pt[0] + r, pt[1] + f), (0, 255, 0), 3)
        if pt[0] < 300 and pt[1]<200:
            m1 = [pt[0],pt[1]]
        if pt[0] > 300 and pt[0] < 1600 and pt[1]<200:
            m2 = [pt[0],pt[1]]
        if pt[0] < 300 and pt[1] > 2000:
            m3 = [pt[0],pt[1]]
        if pt[0] > 300 and pt[0] < 1600 and pt[1]>2000:
            m4 = [pt[0],pt[1]]
        if pt[0] > 1600 and pt[0] < 3000 and pt[1]<200:
            m5 = [pt[0],pt[1]]
        if pt[0] > 3000 and pt[1]<200:
            m6 = [pt[0],pt[1]]
        if pt[0] > 1600 and pt[0] < 3000 and pt[1]>2000:
            m7 = [pt[0],pt[1]]
        if pt[0] > 3000 and pt[1]>2000:
            m8 = [pt[0],pt[1]]
    
    #list the X and Y for each page
    lx1 = [m1[0],m2[0],m3[0],m4[0]]
    ly1 = [m1[1],m2[1],m3[1],m4[1]]
    lx2 = [m5[0],m6[0],m7[0],m8[0]]
    ly2 = [m5[1],m6[1],m7[1],m8[1]]

    #compare their values to identify and correct leaning
    if m1[1] > m2[1]:
        dy = m1[1]-m2[1]
        dx = m2[0]-m1[0]
        arc = np.degrees(np.arctan(dy/dx))
        #print(arc)
        rows, cols, j = img.shape
        M = cv2.getRotationMatrix2D((cols//2,rows//2),-arc,1)
        imgf = cv2.warpAffine(img,M,(cols,rows))
    elif m2[1] > m1[1]:
        dy = m2[1]-m1[1]
        dx = m2[0]-m1[0]    
        arc = np.degrees(np.arctan(dy/dx))
        #print(arc)
        rows, cols, j = img.shape
        M = cv2.getRotationMatrix2D((cols//2,rows//2),arc,1)
        imgf = cv2.warpAffine(img,M,(cols,rows))
    elif m1[1] == m2[1]:
        imgf = img

    if m7[1]>m8[1]:
        dy2 = m7[1] - m8[1]
        dx2 = m8[0] - m7[0]
        arc2 = np.degrees(np.arctan(dy2/dx2))
        #print(arc2)
        rows, cols, j = img.shape
        M = cv2.getRotationMatrix2D((cols//2,rows//2),-arc2,1)
        imgf2 = cv2.warpAffine(img,M,(cols,rows))
    elif m8[1]>m7[1]:
        dy2 = m8[1]-m7[1]
        dx2 = m8[0]-m7[0]    
        arc2 = np.degrees(np.arctan(dy2/dx2))
        #print(arc2)
        rows, cols, j = img.shape
        M = cv2.getRotationMatrix2D((cols//2,rows//2),arc2,1)
        imgf2 = cv2.warpAffine(img,M,(cols,rows))
    elif m8[1] == m7[1]:
        imgf2 = img
    #Crop the new images
    img1 = imgf[ min(ly1, key=int)+int(h/2): max(ly1, key=int),  min(lx1, key=int)+int(w/1.5): max(lx1, key=int)+int(w/3.5)]
    img2 = imgf2[ min(ly2, key=int)+int(h/2): max(ly2, key=int)+int(h/2),  min(lx2, key=int)+int(w/2): max(lx2, key=int)+int(w/2)]

    return img1, img2
