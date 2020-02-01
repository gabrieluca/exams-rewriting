import cv2
import numpy as np

from Normalize_and_crop import normalize_crop

img = cv2.imread("database_files/exams/NM7510_P3_B_2019_2/NM7510_P3_pacote_B_2_sem_2019-02.png",1) 
template = cv2.imread('database_files/track_markers/template.png',cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread('database_files/track_markers/template2.png',cv2.IMREAD_GRAYSCALE)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
# Remove horizontal
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(img, [c], -1, (255,255,255),3)

# Repair image
repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)



def main():
 
    img1, img2 = normalize_crop(img,template,template2)
    img_qdr = img1[0:300, 0:1900]
    img_res = img_qdr[5:500, 40:1900]
    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    # Normalize and threshold image
    res, im = cv2.threshold(img_res, 166, 255, cv2.THRESH_BINARY_INV)

    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(im, None, (0, 0), 255)

    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(im, None, (0, 0), 0)

    imgFI = cv2.bitwise_not(im)
  
    img_qdr = cv2.resize(img_qdr, (800, 200))
    cv2.imshow("Dados do cabecalho:", img_qdr)


        
    cv2.imshow("RA-1)", imgFI[80:150,230:290])
    cv2.waitKey(0)
    cv2.imshow("RA-2)", imgFI[80:150,290:350])
    cv2.waitKey(0)
    cv2.imshow("RA-3)", imgFI[80:150,390:450])
    cv2.waitKey(0)
    cv2.imshow("RA-4)", imgFI[80:150,460:520])
    cv2.waitKey(0)
    cv2.imshow("RA-5)", imgFI[80:150,530:590])
    cv2.waitKey(0)
    cv2.imshow("RA-6)", imgFI[80:150,620:680])
    cv2.waitKey(0)
    cv2.imshow("RA-7)", imgFI[80:150,700:760])
    cv2.waitKey(0)
    cv2.imshow("RA-8)", imgFI[80:150,770:830])
    cv2.waitKey(0)
    cv2.imshow("RA-9)", imgFI[80:150,860:920])
    cv2.waitKey(0)
    cv2.imshow("Sequencial-1)", imgFI[80:150,1000:1070])
    cv2.waitKey(0)
    cv2.imshow("Sequencial-2)", imgFI[80:150,1070:1140])
    cv2.waitKey(0)
    cv2.imshow("Sequencial-3)", imgFI[80:150,1140:1210])
    cv2.waitKey(0)
    cv2.imshow("Pacote)", imgFI[80:150,1280:1350])
    cv2.waitKey(0)

    #img1, img2 = normalize_crop(img,template,template2)
    img_qdr = img1[170:280, 0:1900] #Nome e ass
    img_res = img_qdr[5:500, 40:1900]
    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    # Normalize and threshold image
    res, im = cv2.threshold(img_res, 166, 255, cv2.THRESH_BINARY_INV)

    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(im, None, (0, 0), 255)

    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(im, None, (0, 0), 0)

    imgFI = cv2.bitwise_not(im)

    #img_qdr = cv2.resize(img_qdr, (900, 120))  
    #cv2.imshow("Dados do cabecalho:", img_qdr)

    cv2.imshow("Nome)", imgFI[0:280,80:700])
    cv2.waitKey(0)
    cv2.imshow("Assinatura)", imgFI[0:280,700:1500])
    cv2.waitKey(0)
        
main()





