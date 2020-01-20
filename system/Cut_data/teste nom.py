import cv2
import numpy as np

img = cv2.imread("database_files/exams/NM7510_P3_C_2019_2/NM7510_P3_pacote_C_2_sem_2019-26.png",1) 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('database_files/track_markers/template1.png',cv2.IMREAD_GRAYSCALE)
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

if pt [0]<1600:
    print("{},{},{},{}".format(rt[0],rt[1],pt[0],pt[1]))
    img = img[pt[0]+int(w*1.5):pt[0]+340, pt[0]+w:rt[0]]

    Nome=img [162:220, 76:624]
    cv2.imshow("Nome", Nome)
    cv2.waitKey(0)

       #Salvar Imagem
    cv2.imwrite("Nome.jpg", Nome)

else:
    if rt[0]<1650:

        print("{},{},{},{}".format(rt[0],rt[1],pt[0],pt[1]))
    img = img[pt[0]+int(w*1.5)-1650:pt[0]+340-1630, pt[0]+w-1650:rt[0]-1630]


    Nome=img [175:238, 91:650]
    cv2.imshow("Nome", Nome)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Nome.jpg", Nome)


cv2.destroyAllWindows()