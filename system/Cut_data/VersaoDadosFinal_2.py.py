import cv2
import numpy as np

img = cv2.imread("database_files/exams/NM7510_P3_A_2019_2/NM7510_P3_pacote_A_2_sem_2019-21.png",1) 
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

    RA1=img [70:105, 200:920]
    cv2.imshow("RA-1", RA1)
    cv2.waitKey(0)

    Sequencial=img [50:110, 950:1220]
    cv2.imshow("Sequencial", Sequencial)
    cv2.waitKey(0)

    Pacote=img [50:110, 1280:1339]
    cv2.imshow("Pacote", Pacote)
    cv2.waitKey(0)

    Nome=img [161:237, 74:550]
    cv2.imshow("Nome", Nome)
    cv2.waitKey(0)

    Assinatura=img [152:228, 842:1350]
    cv2.imshow("Assinatura", Assinatura)
    cv2.waitKey(0)

else:
    if rt[0]<1650:

        print("{},{},{},{}".format(rt[0],rt[1],pt[0],pt[1]))
    img = img[pt[0]+int(w*1.5)-1650:pt[0]+340-1630, pt[0]+w-1650:rt[0]-1630]

     if 
    RA=img [68:137, 200:950]
    cv2.imshow("RA", RA)
    cv2.waitKey(0)

    else:
        if 
        RA=img [64:126, 200:950]
        cv2.imshow("RA", RA)
        cv2.waitKey(0)


    #RA=img [66:127, 200:950]
    #Salvar Imagem
    cv2.imwrite("Numero RA.jpg", RA)

    #Sequencial=img [68:130, 980:1250]
    cv2.imshow("Sequencial", Sequencial)
    cv2.waitKey(0)

    #Pacote=img [68:130, 1280:1380]
    cv2.imshow("Pacote", Pacote)
    cv2.waitKey(0)

    #Nome=img [171:241, 93:650]
    cv2.imshow("Nome", Nome)
    cv2.waitKey(0)

   # Assinatura=img [172:242, 870:1410]
    cv2.imshow("Assinatura", Assinatura)
    cv2.waitKey(0)

cv2.destroyAllWindows()