import cv2
import numpy as np

img = cv2.imread("database_files/exams/NM7510_P3_B_2019_2/NM7510_P3_pacote_B_2_sem_2019-54.png",1) 

template = cv2.imread('database_files/track_markers/template3.png',cv2.IMREAD_GRAYSCALE)

def main():
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    print('{0}, {1}, {2}, {3}'.format(pt[0],pt[1],rt[0],rt[1]))

    if pt[0]>1600:
        img1 = img[int(pt[0]/50)+int(w*1.8):rt[1] + int(f/2.5), pt[0]+int(w*0.7):rt[0]+int(w*0.5)]
        img2 = img1[1760:2300, 780:1710]
        img3 = img2[50:400, 335:560]
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

        # Normalize and threshold image
        res, im = cv2.threshold(img3, 165, 255, cv2.THRESH_BINARY_INV)

        # Fill everything that is the same colour (black) as top-left corner with white
        cv2.floodFill(im, None, (0, 0), 255)

        # Fill everything that is the same colour (white) as top-left corner with black
        cv2.floodFill(im, None, (0, 0), 0)

        imgFI = cv2.bitwise_not(im)

        cv2.imshow("Quadro respostas:", img2)

        cv2.imshow("a)", imgFI[0:60,])
        cv2.waitKey(0)
        cv2.imshow("b)", imgFI[55:120,])
        cv2.waitKey(0)
        cv2.imshow("c)", imgFI[115:190,])
        cv2.waitKey(0)
        cv2.imshow("d)", imgFI[185:260,])
        cv2.waitKey(0)
        cv2.imshow("e)", imgFI[250:,])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else: 
        if rt[0]<1650:
            img1 = img[int(pt[0]/50)+int(w*1.95):rt[1] + int(f*0.5), pt[0]+1680:rt[0]+w+1619]
            img2 = img1[1760:2180, 790:1710]
            img3 = img2[65:365, 105:590]
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

            # Normalize and threshold image
            res, im = cv2.threshold(img3, 165, 255, cv2.THRESH_BINARY_INV)

            # Fill everything that is the same colour (black) as top-left corner with white
            cv2.floodFill(im, None, (0, 0), 255)

            # Fill everything that is the same colour (white) as top-left corner with black
            cv2.floodFill(im, None, (0, 0), 0)

            imgFI = cv2.bitwise_not(im)

            cv2.imshow("Quadro respostas:", img2)
            cv2.imshow("img3", img3)

            cv2.imshow("a1)", imgFI[0:60,90:240])
            cv2.waitKey(0)
            cv2.imshow("a2)", imgFI[0:57,240:])
            cv2.waitKey(0)
            cv2.imshow("b)", imgFI[55:110,240:])
            cv2.waitKey(0)
            cv2.imshow("c)", imgFI[110:165,240:])
            cv2.waitKey(0)
            cv2.imshow("d)", imgFI[160:225,240:])
            cv2.waitKey(0)
            cv2.imshow("e)", imgFI[225:,240:])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            img1 = img[int(pt[0]/50)+int(w*2.2):rt[1] + int(f*0.5), pt[0]+1687:rt[0]+int(w*0.5)]
            img2 = img1[1760:2180, 790:1710]
            img3 = img2[65:365, 105:590]
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

            # Normalize and threshold image
            res, im = cv2.threshold(img3, 165, 255, cv2.THRESH_BINARY_INV)

            # Fill everything that is the same colour (black) as top-left corner with white
            cv2.floodFill(im, None, (0, 0), 255)

            # Fill everything that is the same colour (white) as top-left corner with black
            cv2.floodFill(im, None, (0, 0), 0)

            imgFI = cv2.bitwise_not(im)

            cv2.imshow("Quadro respostas:", img2)
            cv2.imshow("img3", img3)

            cv2.imshow("a1)", imgFI[0:60,90:240])
            cv2.waitKey(0)
            cv2.imshow("a2)", imgFI[0:57,240:])
            cv2.waitKey(0)
            cv2.imshow("b)", imgFI[55:110,240:])
            cv2.waitKey(0)
            cv2.imshow("c)", imgFI[110:165,240:])
            cv2.waitKey(0)
            cv2.imshow("d)", imgFI[160:235,240:])
            cv2.waitKey(0)
            cv2.imshow("e)", imgFI[235:,240:])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

main()