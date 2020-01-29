import cv2
import numpy as np

img = cv2.imread("database_files/exams/NM7510_P3_A_2019_2/NM7510_P3_pacote_A_2_sem_2019-01.png",1) 

template = cv2.imread('database_files/track_markers/template1.png',cv2.IMREAD_GRAYSCALE)


def main():
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    
    print("m1: [{0},{1}]\nm2: [{2},{3}]\nm3: [{4},{5}]\nm4: [{6},{7}]".format(m1x, m1y, m2x, m2y, m3x, m3y, m4x, m4y))

    lx1 = [m1x,m2x,m3x,m4x]
    ly1 = [m1y,m2y,m3y,m4y]

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

    img1 = imgf[ min(ly1, key=int)+int(h/2): max(ly1, key=int),  min(lx1, key=int)+int(w/1.5): max(lx1, key=int)+int(w/3.5)]

    #img1 = img1[::3,::3]
    #cv2.imshow("img", img1)
    #cv2.waitKey(0)

    print('{0}, {1}, {2}, {3}'.format(pt[0],pt[1],rt[0],rt[1]))

    if pt[0]>1600:
        
        img1 = img[pt[0]+int(w*1.5)-1650:pt[0]+340-1630, pt[0]+w-1650:rt[0]-1630]
        img2 = img1[0:540, 80:1710]
        img3 = img2[0:200, 150:1300]
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

        # Normalize and threshold image
        res, im = cv2.threshold(img3, 165, 255, cv2.THRESH_BINARY_INV)

        # Fill everything that is the same colour (black) as top-left corner with white
        cv2.floodFill(im, None, (0, 0), 255)

        # Fill everything that is the same colour (white) as top-left corner with black
        cv2.floodFill(im, None, (0, 0), 0)

        imgFI = cv2.bitwise_not(im)

        img2 = cv2.resize(img2, (800, 200))
        cv2.imshow("Dados:", img2)


        cv2.imshow("Sequencial-1)", imgFI[66:133,780:850])
        cv2.waitKey(0)
        cv2.imshow("Sequencial-2)", imgFI[66:133,850:920])
        cv2.waitKey(0)
        cv2.imshow("Sequencial-3)", imgFI[66:133,920:990])
        cv2.waitKey(0)
        cv2.imshow("Pacote)", imgFI[66:133,1060:1130])
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        
main()









