import os
import cv2
import glob
import shutil
import re


def rotate_image(mat, angle):

    height, width, _ = mat.shape
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


class PDFJPGConvert:
    def __init__(self, pdf_path, image_folder):
        self.path = pdf_path
        self.folder = image_folder

    def main_convert(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        with open(self.path, "rb") as file:
            file.seek(0)
            pdf = file.read()

        start_mark = b"\xff\xd8"
        start_fix = 0
        end_mark = b"\xff\xd9"
        end_fix = 2
        i = 0

        n_jpg = 0
        while True:
            i_stream = pdf.find(b"stream", i)
            if i_stream < 0:
                break
            i_start = pdf.find(start_mark, i_stream, i_stream + 20)
            if i_start < 0:
                i = i_stream + 20
                continue
            i_end = pdf.find(b"endstream", i_start)
            if i_end < 0:
                raise Exception("Didn't find end of stream!")
            i_end = pdf.find(end_mark, i_end - 20)
            if i_end < 0:
                raise Exception("Didn't find end of JPG!")

            i_start += start_fix
            i_end += end_fix
            print("JPG %d from %d to %d" % (n_jpg, i_start, i_end))
            jpg = pdf[i_start:i_end]
            with open(self.folder + "jpg%d.jpg" % n_jpg, "wb") as jpgfile:
                jpgfile.write(jpg)

            n_jpg += 1
            i = i_end
            print(n_jpg)

        # -------------------- Read all images in the specified folder -------------------------
        images = []
        types = '*.jpg'

        for files in types:
            images.extend(glob.glob(os.path.join(self.folder, files)))

        def atoi(text):
            return int(text) if text.isdigit() else text
        
        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)', text)]

        images.sort(key=natural_keys)

        # -------------------- Move the front and back images to their database --------------------
        j = 0
        for i in range(1, len(images)):
            if i % 2 != 0:
                j += 1

            new_path = "ImageDB/" + str(j)

            if not os.path.exists(new_path):
                os.makedirs(new_path)

            shutil.move(images[i], new_path)

        # -------------------- Crop RA_Number, Answer images -----------------------------
        for k in range(j):
            path = "ImageDB/" + str(k+1)
            # Select the first image
            image1 = os.listdir(path)[0]
            img1 = cv2.imread(path + '/' + image1)
            ra_img = img1[1400:2460, 3070:3213, :]
            cv2.imwrite(os.path.join(path, 'RA_Number.png'), ra_img)
            ans1_img = img1[670:1070, 200:850, :]
            cv2.imwrite(os.path.join(path, 'Answer1.png'), ans1_img)
            # Select the second image
            image2 = os.listdir(path)[2]
            img2 = cv2.imread(path + '/' + image2)
            ans2_img = img2[3947:4324, 2535:3210, :]
            cv2.imwrite(os.path.join(path, 'Answer2.png'), ans2_img)

        return j
