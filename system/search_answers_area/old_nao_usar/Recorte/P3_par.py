import cv2
import numpy as np
import os

from Normalize_and_crop import normalize_crop

def main_par(img,template,template2,Exam_pack,Page_number):
    img1, img2 = normalize_crop(img,template,template2)
    img_qdrp = img2[1870:2153, 790:1480]
    ares1 = img1[1458:2138,5:1460]
    ares2 = img2[23:2150,19:1470]
    img_res = img_qdrp[:, 105:570]
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
        os.mkdir("C:/Users/Windows/Documents/GitHub/exams-system/Images/Answers/{0}_{1}".format(Exam_pack, Page_number))
        print("Directory " , Exam_pack ,"_", Page_number ,  " Created ") 
    except FileExistsError:
        print("Directory " , Exam_pack ,"_", Page_number ,  " already exists")

    # Save images in the target Directory
    path = "C:/Users/Windows/Documents/GitHub/exams-system/Images/Answers/{0}_{1}".format(Exam_pack, Page_number)
    cv2.imwrite(os.path.join(path, 'Area_resolucao_folha_1({0}).png'.format(Page_number)), ares1)
    cv2.imwrite(os.path.join(path, 'Area_resolucao_folha_2({0}).png'.format(Page_number)), ares2)
    cv2.imwrite(os.path.join(path, 'a1({0}).png'.format(Page_number)), imgFI[:55,50:232])
    cv2.imwrite(os.path.join(path, 'a2({0}).png'.format(Page_number)), imgFI[:55,232:])
    cv2.imwrite(os.path.join(path, 'b({0}).png'.format(Page_number)), imgFI[55:110,232:])
    cv2.imwrite(os.path.join(path, 'c({0}).png'.format(Page_number)), imgFI[110:165,232:])
    cv2.imwrite(os.path.join(path, 'd({0}).png'.format(Page_number)), imgFI[165:220,232:])
    cv2.imwrite(os.path.join(path, 'e({0}).png'.format(Page_number)), imgFI[220:,232:])

    ares1 = ares1[::2,::2]
    ares2 = ares2[::3,::3]
    cv2.imshow("Area de resolucao folha 1", ares1)
    cv2.imshow("area de resolucao folha 2", ares2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("respostas:", img_res)

    cv2.imshow("a1)", imgFI[:55,50:232])
    cv2.waitKey(0)
    cv2.imshow("a2)", imgFI[:55,232:])
    cv2.waitKey(0)
    cv2.imshow("b)", imgFI[55:110,232:])
    cv2.waitKey(0)
    cv2.imshow("c)", imgFI[110:165,232:])
    cv2.waitKey(0)
    cv2.imshow("d)", imgFI[165:220,232:])
    cv2.waitKey(0)
    cv2.imshow("e)", imgFI[220:,232:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()