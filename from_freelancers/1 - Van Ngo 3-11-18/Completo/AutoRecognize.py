import numpy as np
import cv2
import mnist_test

from scipy.ndimage import rotate
from PDF2JPG import PDFJPGConvert


pdf_path = 'PDF/p1.pdf'
folder = 'Images/'


pdf_jpg = PDFJPGConvert(pdf_path, folder)
students = pdf_jpg.main_convert()


print('Converted all PDF to JPG')


def clear_border(image, radius):

    # Given a black and white image, first find all of its contours
    _, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    rows = image.shape[0]
    cols = image.shape[1]
    
    # ID list of contours that touch the border
    contour_list = []

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            row_cnt = pt[0][1]
            col_cnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = ((row_cnt >= 0) and (row_cnt < radius)) or ((row_cnt >= rows-1-radius) and (row_cnt < rows))
            check2 = ((col_cnt >= 0) and (col_cnt < radius)) or ((col_cnt >= cols-1-radius) and (col_cnt < cols))

            if check1 or check2:
                contour_list.append(idx)
                break

    for idx in contour_list:
        cv2.drawContours(image, contours, idx, (0, 0, 0), -1)

    return image


def main():

    loaded_model = mnist_test.model()

    for i in range(students):

        image_list = ['RA_Number.png', 'Answer1.png', 'Answer2.png']

        for jj in range(3):

            image_path = 'ImageDB/' + str(i + 1) + '/' + image_list[jj]

            image = cv2.imread(image_path)

            image = rotate(image, 90)

            im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Normalize and threshold image
            res, img = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Fill everything that is the same colour (black) as top-left corner with white
            cv2.floodFill(img, None, (0, 0), 255)

            # Fill everything that is the same colour (white) as top-left corner with black
            cv2.floodFill(img, None, (0, 0), 0)

            # morphology operation
            kernel = np.ones((3, 3), np.uint8)
            dilate = cv2.dilate(img, kernel, iterations=1)
            opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            
            # remove border
            closing = clear_border(closing, 5)

            # find all your connected components (white blobs in your image)
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)
            sizes = stats[1:, -1]
            nb_components = nb_components - 1

            # minimum size of particles we want to keep (number of pixels)
            # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
            min_size = 120

            # your answer image
            temp = np.zeros(output.shape, dtype=np.uint8)
            # for every component in the image, you keep it only if it's above min_size
            for j in range(0, nb_components):
                if sizes[j] >= min_size:
                    temp[output == j + 1] = 255

            # find contours
            _, contours, hierarchy = cv2.findContours(temp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # sort contours
            sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

            letter_pos = []

            for k, ctr in enumerate(sorted_contours):

                # Get bounding box
                x, y, w, h = cv2.boundingRect(ctr)

                letter = temp[y:y + h, x:x + w]

                letter_pos.append(h)

                if (k > 0) and (h < 0.7 * letter_pos[0]):
                    ans1 = '.'
                else:
                    roi = cv2.resize(letter, (14, 28), cv2.INTER_AREA)

                    new_image = roi.flatten()

                    new_image = new_image.reshape(1, 1, 28, 14)

                    ans1 = loaded_model.predict(new_image)
                    ans1 = ans1.tolist()
                    ans1 = ans1[0].index(max(ans1[0]))

                cv2.putText(image, str(ans1), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('img', image)
                cv2.waitKey(0)


main()
