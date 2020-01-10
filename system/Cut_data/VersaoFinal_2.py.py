import cv2
import numpy as np

img = cv2.imread("database_files/exams/NM7510_P3_2019_2/NM7510_P3_pacote_A_2_sem_2019-19.png",1) 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('database_files/track_markers/template3.png',cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
loc = np.where(result >= 0.659)

for rt in zip(*loc[::-1]):
    cv2.rectangle(img, rt, (rt[0] + w, rt[1] + h), (0, 255, 0), 3)

template2 = cv2.imread('database_files/track_markers/template4.png',cv2.IMREAD_GRAYSCALE)
r, f = template2.shape[::-1]

result2 = cv2.matchTemplate(gray_img, template2, cv2.TM_CCOEFF_NORMED)
loc2 = np.where(result2 >= 0.659)

for pt in zip(*loc2[::-1]):
    cv2.rectangle(img, pt, (pt[0] + r, pt[1] + f), (0, 255, 0), 3)

if pt [0]<1600:
    print("{},{},{},{}".format(rt[0],rt[1],pt[0],pt[1]))
    img = img[pt[0]+int(w*1.5):pt[0]+340, pt[0]+w:rt[0]]

    RA=img [15:135, 200:920]
    cv2.imshow("RAP", RA)
    cv2.waitKey(0)

    Sequencial=img [15:125, 950:1220]
    cv2.imshow("SequencialP", Sequencial)
    cv2.waitKey(0)

    Pacote=img [10:135, 1250:1380]
    cv2.imshow("PacoteP", Pacote)
    cv2.waitKey(0)

    Nome=img [130:250, 60:640]
    cv2.imshow("NomeP", Nome)
    cv2.waitKey(0)

    Assinatura=img [120:245, 825:1420]
    cv2.imshow("AssinaturaP", Assinatura)
    cv2.waitKey(0)

else:
    if rt[0]<1650:

        print("{},{},{},{}".format(rt[0],rt[1],pt[0],pt[1]))
    img = img[pt[0]+int(w*1.5)-1650:pt[0]+340-1630, pt[0]+w-1650:rt[0]-1630]

    RA=img [45:150, 200:950]
    cv2.imshow("RAI", RA)
    cv2.waitKey(0)

    Sequencial=img [45:150, 980:1250]
    cv2.imshow("SequencialI", Sequencial)
    cv2.waitKey(0)

    Pacote=img [45:150, 1280:1380]
    cv2.imshow("PacoteI", Pacote)
    cv2.waitKey(0)

    Nome=img [155:265, 75:670]
    cv2.imshow("NomeI", Nome)
    cv2.waitKey(0)

    Assinatura=img [155:265, 850:1440]
    cv2.imshow("AssinaturaI", Assinatura)
    cv2.waitKey(0)

cv2.destroyAllWindows()