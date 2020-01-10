import cv2
import numpy as np

Imagem = cv2.imread("database_files/exams/NM7510_P3_2019_2/NM7510_P3_pacote_A_2_sem_2019-01.png",1)

#Recorte da região de interesse (RA)
RA=Imagem[160:235, 265:815]

#Exibir Imagem
cv2.imshow("RA", RA)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Salvar Imagem
#cv2.imwrite("Numero RA.jpg", RA)

#Recorte da região de interesse (Sequencial)
Sequencial=Imagem[160:235, 845:1025]

#Exibir Imagem
cv2.imshow("Sequencial", Sequencial)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Salvar Imagem
#cv2.imwrite("Numero sequencial.jpg", Sequencial)

#Recorte da região de interesse (Pacote)
Pacote=Imagem[160:235, 1065:1130]

#Exibir Imagem
cv2.imshow("Pacote", Pacote)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Salvar Imagem
#cv2.imwrite("Pacote.jpg", Pacote)

#Recorte da região de interesse (Nome)
Nome=Imagem[245:320, 155:595]

#Exibir Imagem
cv2.imshow("Nome", Nome)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Salvar Imagem
#cv2.imwrite("Nome.jpg", Nome)

#Recorte da região de interesse (Assinatura)
Assinatura=Imagem[245:320, 740:1170]

#Exibir Imagem
cv2.imshow("Assinatura", Assinatura)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Salvar Imagem
#cv2.imwrite("Assinatura.jpg", Assinatura)
