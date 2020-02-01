import cv2
import numpy as np

img = cv2.imread("database_files/exams/NM7510_P3_A_2019_2/NM7510_P3_pacote_A_2_sem_2019-04.png",1) 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('database_files/track_markers/template.png',cv2.IMREAD_GRAYSCALE)
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

if pt [0]<1600: #Par
    print("{},{},{},{}".format(rt[0],rt[1],pt[0],pt[1]))
    img = img[pt[0]+int(w*1.5):pt[0]+340, pt[0]+w:rt[0]]

    RA1=img [43:102, 223:278]
    cv2.imshow("RA-1", RA1)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA1.jpg", RA1)

    RA2=img [42:102, 293:348]
    cv2.imshow("RA-2", RA2)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA2.jpg", RA2)

    RA3=img [41:102, 379:435]
    cv2.imshow("RA-3", RA3)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA3.jpg", RA3)

    RA4=img [40:102, 453:505]
    cv2.imshow("RA-4", RA4)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA4.jpg", RA4)

    RA5=img [40:100, 523:575]
    cv2.imshow("RA-5", RA5)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA5.jpg", RA5)

    RA6=img [40:100, 610:660]
    cv2.imshow("RA-6", RA6)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA6.jpg", RA6)

    RA7=img [40:100, 690:735]
    cv2.imshow("RA-7", RA7)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA7.jpg", RA7)

    RA8=img [40:98, 753:800]
    cv2.imshow("RA-8", RA8)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA8.jpg", RA8)

    RA9=img [40:98, 853:905]
    cv2.imshow("RA-9", RA9)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA9.jpg", RA9)

    Sequencial1=img [40:98, 998:1050]
    cv2.imshow("Sequencial-1", Sequencial1)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Sequencial1.jpg", Sequencial1)

    Sequencial2=img [40:98, 1065:1120]
    cv2.imshow("Sequencial-2", Sequencial2)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Sequencial2.jpg", Sequencial2)

    Sequencial3=img [40:95, 1135:1190]
    cv2.imshow("Sequencial-3", Sequencial3)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Sequencial3.jpg", Sequencial3)

    Pacote=img [40:95, 1280:1335]
    cv2.imshow("Pacote", Pacote)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Pacote.jpg", Pacote)

    Nome=img [145:215, 73:625]
    cv2.imshow("Nome", Nome)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Nome.jpg", Nome)

    Assinatura=img [135:205, 840:1375]
    cv2.imshow("Assinatura", Assinatura)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Assinatura.jpg", Assinatura)

else:
    if rt[0]<1650: #Impar

        print("{},{},{},{}".format(rt[0],rt[1],pt[0],pt[1]))
    img = img[pt[0]+int(w*1.5)-1650:pt[0]+340-1630, pt[0]+w-1650:rt[0]-1630]

 
    RA1=img [66:123, 240:298]
    cv2.imshow("RA-1", RA1)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA1.jpg", RA1)

    RA2=img [66:123, 313:370]
    cv2.imshow("RA-2", RA2)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA2.jpg", RA2)
 
    RA3=img [66:123, 400:460]
    cv2.imshow("RA-3", RA3)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA3.jpg", RA3)

    RA4=img [66:123, 473:530]
    cv2.imshow("RA-4", RA4)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA4.jpg", RA4)

    RA5=img [66:123, 545:603]
    cv2.imshow("RA-5", RA5)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA5.jpg", RA5)

    RA6=img [66:123, 633:688]
    cv2.imshow("RA-6", RA6)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA6.jpg", RA6)

    RA7=img [66:123, 705:760]
    cv2.imshow("RA-7", RA7)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA7.jpg", RA7)

    RA8=img [66:123, 705:760]
    cv2.imshow("RA-8", RA8)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA8.jpg", RA8)

    RA9=img [66:123, 775:830]
    cv2.imshow("RA-9", RA9)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("RA9.jpg", RA9)

    Sequencial1=img [66:123, 1023:1075]
    cv2.imshow("Sequencial-1", Sequencial1)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Sequencial1.jpg", Sequencial1)

    Sequencial2=img [66:123, 1090:1147]
    cv2.imshow("Sequencial-2", Sequencial2)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Sequencial2.jpg", Sequencial2)

    Sequencial3=img [66:123, 1163:1220]
    cv2.imshow("Sequencial-3", Sequencial3)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Sequencial3.jpg", Sequencial3)

    Pacote=img [66:123, 1305:1360]
    cv2.imshow("Pacote", Pacote)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Pacote.jpg", Pacote)

    Nome=img [173:241, 93:625]
    cv2.imshow("Nome", Nome)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Nome.jpg", Nome)

    Assinatura=img [175:242, 870:1375]
    cv2.imshow("Assinatura", Assinatura)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Assinatura.jpg", Assinatura)

cv2.destroyAllWindows()