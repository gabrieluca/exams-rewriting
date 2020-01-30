import cv2
import numpy as np

from Normalize_and_crop import normalize_crop

img = cv2.imread("database_files/exams/NM7510_P3_C_2019_2/NM7510_P3_pacote_C_2_sem_2019-04.png",1) 
template = cv2.imread('database_files/track_markers/template.png',cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread('database_files/track_markers/template2.png',cv2.IMREAD_GRAYSCALE)

def main():
    img1, img2 = normalize_crop(img,template,template2)
    img_qdr = img2[1815:2155, 790:1466]
    img_res = img_qdr[:, 135:570]
    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    # Normalize and threshold image
    res, im = cv2.threshold(img_res, 166, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

     # morphology operation
    kernel = np.ones((3, 3), np.uint8)
    im = cv2.dilate(im, kernel, iterations=1)

    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 80

    # your answer image
    temp = np.zeros(output.shape, dtype=np.uint8)
    # for every component in the image, you keep it only if it's above min_size
    for j in range(0, nb_components):
        if sizes[j] >= min_size:
            temp[output == j + 1] = 255
    temp = cv2.bitwise_not(temp)

    im2, contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    questoes = ['e)','d)','c)','b)','a)','a2)']
    f=0

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        resp = im[ y : y+h , x : x+w ]
        a,b = resp.shape
        if a < 30 or b < 150:
            d=0
        elif a>40 and b>200 :
            if x+w < 400 and y+h > 115 or f>=6:
                d=0
            else:
                resp = im[ y+2 : y+h-2 , x+2 : x+w-2 ]
                resp = cv2.bitwise_not(resp)
                cv2.imshow(questoes[0+f],resp)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                f+=1

main()