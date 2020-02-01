import cv2
import os
import numpy as np

from Normalize_and_crop import normalize_crop

img = cv2.imread("database_files/exams/NM7510_P3_C_2019_2/NM7510_P3_pacote_C_2_sem_2019-04.png",1) 
template = cv2.imread('database_files/track_markers/template.png',cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread('database_files/track_markers/template2.png',cv2.IMREAD_GRAYSCALE)


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
# Remove horizontal
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(img, [c], -1, (255,255,255),2,5)

# Repair image
repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)


def main():

    

    img1, img2 = normalize_crop(img,template,template2)
    img_qdri = img1[10:300, 10:1780] 
    ares1 = img1[0:300,0:1780]
    ares2 = img1[169:280, 0:1780]
    img_res = img_qdri[5:500, 40:1780]
    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(img_res,190, 255, cv2.THRESH_BINARY)

    # Normalize and threshold image
    res, im = cv2.threshold(img_res, 164, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(blackAndWhiteImage, None, (0, 0), 255)

    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(im, None, (0, 0), 0)

    # morphology operation
    kernel = np.ones((2, 2), np.uint8)
    im = cv2.dilate(im, kernel, iterations=1)

    imgFI = cv2.bitwise_not(im)

   
   
    # Show images
    #blackAndWhiteImage = blackAndWhiteImage[::1,::1]
    ares1 = ares1[::2,::2]
    #blackAndWhiteImage = cv2.resize(blackAndWhiteImage, (900, 200))
    #cv2.imshow("Dados_do_prova", blackAndWhiteImage)
    cv2.imshow("Dados_do_aluno", ares1)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imshow("respostas:", img_res)

    cv2.imshow("RA-1)", imgFI[70:137,228:285])
    cv2.waitKey(0)
    cv2.imshow("RA-2)", imgFI[70:137,298:355])
    cv2.waitKey(0)
    cv2.imshow("RA-3)", imgFI[70:137,388:445])
    cv2.waitKey(0)
    cv2.imshow("RA-4)", imgFI[70:137,458:515])
    cv2.waitKey(0)
    cv2.imshow("RA-5)", imgFI[70:137,528:585])
    cv2.waitKey(0)
    cv2.imshow("RA-6)", imgFI[70:137,615:672])
    cv2.waitKey(0)
    cv2.imshow("RA-7)", imgFI[70:137,685:745])
    cv2.waitKey(0)
    cv2.imshow("RA-8)", imgFI[70:137,758:815])
    cv2.waitKey(0)
    cv2.imshow("RA-9)", imgFI[70:137,860:920])
    cv2.waitKey(0)
    
    cv2.imshow("Sequencial-1)", imgFI[70:137,1005:1050])
    cv2.waitKey(0)
    cv2.imshow("Sequencial-2)", imgFI[70:137,1075:1130])
    cv2.waitKey(0)
    cv2.imshow("Sequencial-3)", imgFI[70:137,1145:1200])
    cv2.waitKey(0)
    cv2.imshow("Pacote)", imgFI[70:137,1290:1345])
    cv2.waitKey(0)


    #cv2.imshow("RA-1)", imgFI[70:137,230:930])
    #cv2.waitKey(0)
    c#v2.imshow("Sequencial-1)", imgFI[70:137,1000:1400])
    #cv2.waitKey(0)


    ares2 = ares2[::2,::2]
    #blackAndWhiteImage = cv2.resize(blackAndWhiteImage, (900, 200))
    #cv2.imshow("Dados_do_prova", blackAndWhiteImage)
    cv2.imshow("Nome_do_aluno", ares2)
    cv2.waitKey(0)

    cv2.imshow("Nome)", imgFI[160:245,82:625])
    cv2.waitKey(0)
    cv2.imshow("Assinatura)", imgFI[160:245,852:1390])
    cv2.waitKey(0)
 

main()