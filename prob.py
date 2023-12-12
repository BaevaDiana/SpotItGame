import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import os

def process_and_save_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            imgname = os.path.splitext(filename)[0]
            image = cv2.imread(os.path.join(input_folder, filename))
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            resized = cv2.resize(final, (800, 800))
            # cv2.imwrite(os.path.join(output_folder, f'{imgname}_processed.jpg'), resized)

            image = resized
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            output = image.copy()

            for c in cnts:
                cv2.drawContours(output, [c], -1, (255, 0, 0), 3)

            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            mask = np.zeros(gray.shape, np.uint8)
            mask = cv2.drawContours(mask, [cnts], -1, 255, cv2.FILLED)
            fg_masked = cv2.bitwise_and(image, image, mask=mask)
            mask = cv2.bitwise_not(mask)
            bk = np.full(image.shape, 255, dtype=np.uint8)
            bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
            final = cv2.bitwise_or(fg_masked, bk_masked)

            gray = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
            thresh = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.bitwise_not(thresh)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

            i = 0
            for c in cnts:
                if cv2.contourArea(c) > 1000:
                    mask = np.zeros(gray.shape, np.uint8)
                    mask = cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
                    fg_masked = cv2.bitwise_and(image, image, mask=mask)
                    mask = cv2.bitwise_not(mask)
                    bk = np.full(image.shape, 255, dtype=np.uint8)
                    bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
                    finalcont = cv2.bitwise_or(fg_masked, bk_masked)

                    output = finalcont.copy()
                    x, y, w, h = cv2.boundingRect(c)

                    if w < h:
                        x += int((w - h) / 2)
                        w = h
                    else:
                        y += int((h - w) / 2)
                        h = w

                    roi = finalcont[y:y + h, x:x + w]

                    # Проверка наличия изображения и ненулевых размеров
                    if not roi.size == 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                        roi = cv2.resize(roi, (400, 400))

                        icon_folder = os.path.join(output_folder, 'symbols', imgname)
                        if not os.path.exists(icon_folder):
                            os.makedirs(icon_folder)

                        cv2.imwrite(os.path.join(icon_folder, f"simb_{i}.jpg"), roi)
                        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        plt.title('Contours')
                        plt.show()
                        i += 1

                    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
                    plt.title('Contours')
                    plt.show()

# Пример использования
input_folder = "f"
output_folder = "output"
process_and_save_images(input_folder, output_folder)
