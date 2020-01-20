import cv2
import numpy as np

img = cv2.imread("database_files/exams/NM7510_P3_A_2019_2/NM7510_P3_pacote_A_2_sem_2019-16.png",1) 
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

    RA=img [79:149, 200:950] #pag 1  
    RA=img [45:100, 200:920] #pag 4 , 8
    RA=img [52:120, 200:920] #pag 2, 6 
    cv2.imshow("RA", RA)
    cv2.waitKey(0)

    #Salvar Imagem
    cv2.imwrite("Numero RA.jpg", RA)

else:
    if rt[0]<1650:

        print("{},{},{},{}".format(rt[0],rt[1],pt[0],pt[1]))
    img = img[pt[0]+int(w*1.5)-1650:pt[0]+340-1630, pt[0]+w-1650:rt[0]-1630]

    
    RA=img [66:135, 200:950] #pag 
    RA=img [66:129, 200:950] #pag 5, 7,3 
    cv2.imshow("RA", RA)
    cv2.waitKey(0)

#Salvar Imagem
    cv2.imwrite("Numero RA.jpg", RA)

cv2.destroyAllWindows()