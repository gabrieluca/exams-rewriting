import cv2
import numpy as np

img = cv2.imread("database_files/exams/Atividades_Primeira_coleta_dados/Atividades_escaneadas (1)-88.png",1) 
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

if pt [0]<1600: #Impar

    print("{},{},{},{}".format(rt[0],rt[1],pt[0],pt[1]))
    img = img[pt[0]+int(w*1.5):pt[0]+340, pt[0]+w:rt[0]]

    #Numero do RA

    RA1=img [45:97, 267:311]
    cv2.imshow("RA1", RA1)
    cv2.waitKey(0)

    RA2=img [45:97, 331:382]
    cv2.imshow("RA2", RA2)
    cv2.waitKey(0)

    RA3=img [45:97, 420:470]
    cv2.imshow("RA3", RA3)
    cv2.waitKey(0)

    RA4=img [45:97, 490:541]
    cv2.imshow("RA4", RA4)
    cv2.waitKey(0)

    RA5=img [45:97, 560:612]
    cv2.imshow("RA5", RA5)
    cv2.waitKey(0)

    RA6=img [45:97, 648:698]
    cv2.imshow("RA6", RA6)
    cv2.waitKey(0)

    RA7=img [45:97, 718:770]
    cv2.imshow("RA7", RA7)
    cv2.waitKey(0)

    RA8=img [45:97, 790:842]
    cv2.imshow("RA8", RA8)
    cv2.waitKey(0)

    RA9=img [45:97, 893:945]
    cv2.imshow("RA9", RA9)
    cv2.waitKey(0)

    #Numero Sequencial

    Sequencial1=img [47:93, 1035:1080]
    cv2.imshow("Sequencial1", Sequencial1)
    cv2.waitKey(0)

    Sequencial2=img [47:93, 1100:1152]
    cv2.imshow("Sequencial2", Sequencial2)
    cv2.waitKey(0)

    Sequencial3=img [47:93, 1172:1222]
    cv2.imshow("Sequencial3", Sequencial3)
    cv2.waitKey(0)

    #Pacote

    Pacote=img [47:93, 1304:1352]
    cv2.imshow("Pacote", Pacote)
    cv2.waitKey(0)

    #Nome

    Nome=img [158:515, 1:725]
    cv2.imshow("Nome", Nome)
    cv2.waitKey(0)

    #Assinatura

    Assinatura=img [163:515, 750:1490]
    cv2.imshow("Assinatura", Assinatura)
    cv2.waitKey(0)
else:
    if rt[0]<1650:

        print("{},{},{},{}".format(rt[0],rt[1],pt[0],pt[1]))
    img = img[pt[0]+int(w*1.5)-1650:pt[0]+340-1630, pt[0]+w-1650:rt[0]-1630]

    Sequencial=img [69:126, 1020:1250] 
    cv2.imshow("Sequencial", Sequencial)
    cv2.waitKey(0)

cv2.destroyAllWindows()