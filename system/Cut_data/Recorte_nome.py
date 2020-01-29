import cv2
import numpy as np

img = cv2.imread("database_files/exams/NM7510_P3_C_2019_2/NM7510_P3_pacote_C_2_sem_2019-01.png",1) 

template = cv2.imread('database_files/track_markers/template1.png',cv2.IMREAD_GRAYSCALE)

def main():
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= 0.659)

    for rt in zip(*loc[::-1]):
        cv2.rectangle(img, rt, (rt[0] + w, rt[1] + h), (0, 255, 0), 3)

    template2 = cv2.imread('database_files/track_markers/template2.png',cv2.IMREAD_GRAYSCALE)
    r, f = template2.shape[::-1]

    result2 = cv2.matchTemplate(gray_img, template2, cv2.TM_CCOEFF_NORMED)
    loc2 = np.where(result2 >= 0.659)

    for pt in zip(*loc2[::-1]):
        cv2.rectangle(img, pt, (pt[0] + r, pt[1] + f), (0, 255, 0), 3)

    

    print('{0}, {1}, {2}, {3}'.format(pt[0],pt[1],rt[0],rt[1]))

    if pt[0]>1600:
        img1 = img[pt[0]+int(w*1.5)-1490:pt[0]+340-1645, pt[0]+w-1685:rt[0]-1610]  #NOME
        img2 = img1[0:540, 0:2100]
        img3 = img2[0:600, 50:1500]
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

        # Normalize and threshold image
        res, im = cv2.threshold(img3, 166, 255, cv2.THRESH_BINARY_INV)

        # Fill everything that is the same colour (black) as top-left corner with white
        cv2.floodFill(im, None, (0, 0), 255)

        # Fill everything that is the same colour (white) as top-left corner with black
        cv2.floodFill(im, None, (0, 0), 0)

        imgFI = cv2.bitwise_not(im)

        img2 = cv2.resize(img2, (800, 100))
        cv2.imshow("Dados:", img2)
 
        cv2.imshow("Nome)", imgFI[0:180,0:1500])
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        


main()