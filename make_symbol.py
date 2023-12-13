import cv2
import imutils
import numpy as np
import os


def process_and_save_images(input_folder, output_folder):
    # Проверяем, существует ли папка вывода, и создаем ее, если нет
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Проходим по каждому файлу в папке ввода
    for filename in os.listdir(input_folder):
        # Проверяем, имеет ли файл допустимое расширение изображения
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            imgname = os.path.splitext(filename)[0]  # Получаем имя изображения без расширения

            # Загружаем изображение
            image = cv2.imread(os.path.join(input_folder, filename))

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

                    x, y, w, h = cv2.boundingRect(c)

                    if w < h:
                        x += int((w - h) / 2)
                        w = h
                    else:
                        y += int((h - w) / 2)
                        h = w

                    roi = finalcont[y:y + h, x:x + w]

                    if not roi.size == 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                        roi = cv2.resize(roi, (400, 400))

                        # Создаем папку для символов, если ее не существует
                        icon_folder = os.path.join(output_folder, 'symbols', imgname)
                        if not os.path.exists(icon_folder):
                            os.makedirs(icon_folder)

                        # Сохраняем обработанный символ
                        cv2.imwrite(os.path.join(icon_folder, f"simb_{i}.jpg"), roi)

# Пример использования
input_folder = "augmented_dataset"
output_folder = "output"
process_and_save_images(input_folder, output_folder)
