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
    img_res = img_qdri[:, 338:570]
    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    # Normalize and threshold image
    res, im = cv2.threshold(img_res, 166, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(im, None, (0, 0), 255)

    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(im, None, (0, 0), 0)

    # morphology operation
    kernel = np.ones((3, 3), np.uint8)
    im = cv2.dilate(im, kernel, iterations=1)

    imgFI = cv2.bitwise_not(im)

    # Create target Directory
    try:
        os.mkdir("C:/Users/Windows/Documents/GitHub/exams-system/Images/Answers/Recorte/{0}_{1}".format(t, x))
        print("Directory ",t,"_",x," Created ") 
    except FileExistsError:
        print("Directory " , t ,"_", x ,  " already exists")

    # Save images in the target Directory
    path = "C:/Users/Windows/Documents/GitHub/exams-system/Images/Answers/Recorte/{0}_{1}".format(t, x)
    cv2.imwrite(os.path.join(path, 'Area_resolucao_folha_1({0}).png'.format(x)), ares1)
    cv2.imwrite(os.path.join(path, 'Area_resolucao_folha_2({0}).png'.format(x)), ares2)
    cv2.imwrite(os.path.join(path, 'a({0}).png'.format(x)), imgFI[:65,:])
    cv2.imwrite(os.path.join(path, 'b({0}).png'.format(x)), imgFI[65:130,:])
    cv2.imwrite(os.path.join(path, 'c({0}).png'.format(x)), imgFI[130:195,:])
    cv2.imwrite(os.path.join(path, 'd({0}).png'.format(x)), imgFI[195:260,:])
    cv2.imwrite(os.path.join(path, 'e({0}).png'.format(x)), imgFI[260:,:])

    # Show images
    ares1 = ares1[::2,::2]
    ares2 = ares2[::3,::3]
    cv2.imshow("Area de resolucao folha 1", ares1)
    cv2.imshow("area de resolucao folha 2", ares2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("respostas:", img_res)

    cv2.imshow("a)", imgFI[:65,:])
    cv2.waitKey(0)
    cv2.imshow("b)", imgFI[65:130,:])
    cv2.waitKey(0)
    cv2.imshow("c)", imgFI[130:195,:])
    cv2.waitKey(0)
    cv2.imshow("d)", imgFI[195:260,:])
    cv2.waitKey(0)
    cv2.imshow("e)", imgFI[260:,:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if y == 0:
    main_par(img,template,template2,t,x)
else:
    main_impar()