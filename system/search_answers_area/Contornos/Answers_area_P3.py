import cv2
import os
import numpy as np

from Normalize_and_crop import normalize_crop
from P3_par import main_par

path = "database_files/exams/NM7510_P3_A_2019_2/NM7510_P3_pacote_A_2_sem_2019-07.png"
template = cv2.imread('database_files/track_markers/template.png',cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread('database_files/track_markers/template2.png',cv2.IMREAD_GRAYSCALE)
img = cv2.imread(path,1)
# Exam Pack
t = str(path[57:58])
# Page number
x = str(path[70:72])
# See if its as even number
y = int(path[70:72]) % 2


def main_impar():
    img1, img2 = normalize_crop(img,template,template2)
    img_qdri = img2[1815:2133, 790:1466]
    ares1 = img1[1591:,:]
    ares2 = img2[23:2130,20:1465]
    img_res = img_qdri[:, 285:570]
    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    # Normalize and threshold image
    res, im = cv2.threshold(img_res, 166, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Fill everything that is the same colour (black) as top-left corner with white
    #cv2.floodFill(im, None, (0, 0), 255)

    # Fill everything that is the same colour (white) as top-left corner with black
    #cv2.floodFill(im, None, (0, 0), 0)

    # morphology operation
    kernel = np.ones((3, 3), np.uint8)
    im = cv2.dilate(im, kernel, iterations=1)

    #imgFI = cv2.bitwise_not(im)

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

    # Find contours
    im2, contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # List the names for each contour
    questoes = ['e)','d)','c)','b)','a)','a2)']
    f=0

    # Create target Directory
    # try:
    #     os.mkdir("C:/Users/Windows/Documents/GitHub/exams-system/Images/Answers/{0}_{1}".format(t, x))
    #     print("Directory ",t,"_",x," Created ") 
    # except FileExistsError:
    #     print("Directory " , t ,"_", x ,  " already exists")

    # # Save images in the target Directory
    # path = "C:/Users/Windows/Documents/GitHub/exams-system/Images/Answers/{0}_{1}".format(t, x)
    # cv2.imwrite(os.path.join(path, 'Area_resolucao_folha_1({0}).png'.format(x)), ares1)
    # cv2.imwrite(os.path.join(path, 'Area_resolucao_folha_2({0}).png'.format(x)), ares2)
    # cv2.imwrite(os.path.join(path, 'a({0}).png'.format(x)), imgFI[:65,:])
    # cv2.imwrite(os.path.join(path, 'b({0}).png'.format(x)), imgFI[65:130,:])
    # cv2.imwrite(os.path.join(path, 'c({0}).png'.format(x)), imgFI[130:195,:])
    # cv2.imwrite(os.path.join(path, 'd({0}).png'.format(x)), imgFI[195:260,:])
    # cv2.imwrite(os.path.join(path, 'e({0}).png'.format(x)), imgFI[260:,:])

    #for each contour analize if it has the min shape and crop the final image
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        resp = im[ y : y+h , x : x+w ]
        a,b = resp.shape
        if a < 30 or b < 150:
            d=0
        elif a>40 and b>200 :
            resp = im[ y+2 : y+h-2 , x+2 : x+w-2 ]
            resp = cv2.bitwise_not(resp)
            cv2.imshow(questoes[0+f],resp)
            cv2.waitKey(0)
            f+=1
    cv2.destroyAllWindows()

    # Show images
    #ares1 = ares1[::2,::2]
    #ares2 = ares2[::3,::3]
    #cv2.imshow("Area de resolucao folha 1", ares1)
    #cv2.imshow("area de resolucao folha 2", ares2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imshow("respostas:", img_res)

    #cv2.imshow("a)", imgFI[:65,:])
    #cv2.waitKey(0)
    #cv2.imshow("b)", imgFI[65:130,:])
    #cv2.waitKey(0)
    #cv2.imshow("c)", imgFI[130:195,:])
    #cv2.waitKey(0)
    #cv2.imshow("d)", imgFI[195:260,:])
    #cv2.waitKey(0)
    #cv2.imshow("e)", imgFI[260:,:])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if y == 0:
    main_par(img,template,template2,t,x)
else:
    main_impar()