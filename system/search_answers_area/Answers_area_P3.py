import cv2
import numpy as np

from Normalize_and_crop import normalize_crop

img = cv2.imread("database_files/exams/NM7510_P3_B_2019_2/NM7510_P3_pacote_B_2_sem_2019-53.png",1) 
template = cv2.imread('database_files/track_markers/template.png',cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread('database_files/track_markers/template2.png',cv2.IMREAD_GRAYSCALE)

def main():
    img1, img2 = normalize_crop(img,template,template2)
    img_qdr = img2[1760:2180, 790:1710]
    img_res = img_qdr[65:365, 105:590]
    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    # Normalize and threshold image
    res, im = cv2.threshold(img_res, 166, 255, cv2.THRESH_BINARY_INV)

    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(im, None, (0, 0), 255)

    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(im, None, (0, 0), 0)

    imgFI = cv2.bitwise_not(im)

    cv2.imshow("Quadro respostas:", img_qdr)

    cv2.imshow("a)", imgFI[0:65,])
    cv2.waitKey(0)
    cv2.imshow("b)", imgFI[65:130,])
    cv2.waitKey(0)
    cv2.imshow("c)", imgFI[125:195,])
    cv2.waitKey(0)
    cv2.imshow("d)", imgFI[195:264,])
    cv2.waitKey(0)
    cv2.imshow("e)", imgFI[264:,])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()