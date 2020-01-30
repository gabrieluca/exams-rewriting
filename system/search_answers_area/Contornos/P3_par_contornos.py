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

    # Create target Directory
    try:
        os.mkdir("C:/Users/Windows/Documents/GitHub/exams-system/Images/Answers/Contour/{0}_{1}".format(Exam_pack, Page_number))
        print("Directory " , Exam_pack ,"_", Page_number ,  " Created ") 
    except FileExistsError:
        print("Directory " , Exam_pack ,"_", Page_number ,  " already exists")
    path = "C:/Users/Windows/Documents/GitHub/exams-system/Images/Answers/Contour/{0}_{1}".format(Exam_pack, Page_number)

    # Save images in the target Directory
    cv2.imwrite(os.path.join(path, 'Area_resolucao_folha_1({0}).png'.format(Page_number)), ares1)
    cv2.imwrite(os.path.join(path, 'Area_resolucao_folha_2({0}).png'.format(Page_number)), ares2)

    # Show images
    ares1 = ares1[::2,::2]
    ares2 = ares2[::3,::3]
    cv2.imshow("Area de resolucao folha 1", ares1)
    cv2.imshow("area de resolucao folha 2", ares2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Normalize and threshold image
    res, im = cv2.threshold(img_res, 166, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # morphology operation
    kernel = np.ones((3, 3), np.uint8)
    im = cv2.dilate(im, kernel, iterations=1)

    img_res2 = im[:, 230:]

    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_res2, connectivity=8)
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
    questoes = ['e','d','c','b','a']
    f=0

    #for each contour analize if it has the min shape and crop the final image
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        resp = img_res2[ y : y+h , x : x+w ]
        a,b = resp.shape
        if a < 30 or b < 150:
            d=0
        elif a>40 and b>200 :
            resp = img_res2[ y+2 : y+h-2 , x+2 : x+w-2 ]
            resp = cv2.bitwise_not(resp)
            cv2.imshow(questoes[0+f],resp)
            cv2.imwrite(os.path.join(path, '{0}({1}).png'.format(questoes[0+f],Page_number)), resp)
            cv2.waitKey(0)
            f+=1

    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(im, None, (0, 0), 255)

    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(im, None, (0, 0), 0)

    imgFI = cv2.bitwise_not(im)

    cv2.imshow("a2({0})".format(Page_number), imgFI[:55,50:232])
    cv2.imwrite(os.path.join(path, 'a2({0}).png'.format(Page_number)), imgFI[:55,50:232])
    cv2.waitKey(0)
    cv2.destroyAllWindows()