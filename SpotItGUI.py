import tkinter as tk
# from PIL import Image, ImageTk
import random
import os
from computervision.opencvutils import Image
from computervision.utils import *
from computervision.kerasutils import *
from computervision.imageioutils import *
from PIL import Image as PILImage, ImageTk

import shutil
import itertools

remove=True

class NeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network App")

        self.start_button = tk.Button(root, text="Сгенерировать", command=self.find_image)
        self.start_button.pack(pady=10)

        self.start_button = tk.Button(root, text="Начать игру", command=self.start_neural_network)
        self.start_button.pack(pady=10)

        self.image_label1 = tk.Label(root)
        self.image_label1.pack(side=tk.LEFT, padx=10)

        self.image_label2 = tk.Label(root)
        self.image_label2.pack(side=tk.LEFT, padx=10)

        self.result_image_label = tk.Label(root)
        self.result_image_label.pack(side=tk.BOTTOM, pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)
        self.images = []
        self.nr=0

    def find_image(self):
        directory = 'dataset_cards'
        images = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

        if len(images) < 2:
            print("Недостаточно изображений в папке.")
            return

        random_images = random.sample(images, 2)
        file_path1 = os.path.join(directory, random_images[0])
        file_path2 = os.path.join(directory, random_images[1])

        # Ваш код для обработки изображений и получения результата

        # Отображение изображений в интерфейсе
        image1 = PILImage.open(file_path1)
        image1 = image1.resize((200, 200))  # Изменение размера изображения
        image1 = ImageTk.PhotoImage(image1)
        self.image_label1.configure(image=image1)
        self.image_label1.image = image1

        image2 = PILImage.open(file_path2)
        image2 = image2.resize((200, 200))  # Изменение размера изображения
        image2 = ImageTk.PhotoImage(image2)

        self.image_label2.configure(image=image2)
        self.image_label2.image = image2

        image2 = Image(file_path2)
        image1 = Image(file_path1)
        self.images=[]
        self.images.append(image1)
        self.images.append(image2)



    def start_neural_network(self):
        # Ваш код для вывода результата
        # result_text = "Result: Symbol X"  # Замените на фактический результат
        # self.result_label.config(text=result_text)

        model = load_cnn_model()
        # images = [Image(path_to_img=join(directory, files[i])) for i in range(len(files))]
        for img in self.images:
            # Обработка изображений для предсказания, добавление контраста, изменение размера, размытие
            img.processed = img.contrast_resized_blurred()
            # Создание директории для предсказаний
            img.preddir = create_prediction_dirs(self.nr)
            # Нахождение самого большого контура: это карта
            cnt = Image.grab_contours_by_area(img.processed)[0]
            img.card = Image.keep_contour_with_white_background(img.processed, cnt)
            # Нахождение 10 самых больших контуров, как минимум, для восьми иконок на карте
            img.cnts = Image.grab_contours_by_area(img.card, reverse=True, threshold=190, area=900)[:10]

            i = 0
            for c in img.cnts:
                # Извлечение прямоугольных координат вокруг контура 'c'
                x, y, w, h = Image.get_rect_coordinates_around_contour(c)

                # Добавление координат (x, y) контура в списки img.cntsx и img.cntsy
                img.cntsx.append(x)
                img.cntsy.append(y)

                # Нанесение контура 'c' на копию обработанного изображения и сохранение результата в img.drawncontour
                img.drawncontour = Image.draw_contour(img.processed.copy(), c)

                # Добавление изображения с нарисованным контуром в список img.cnts_images
                img.cnts_images.append(img.drawncontour)

                # Извлечение контура 'c' и сохранение его на белом фоне из исходного изображения 'img.card'
                img.icon = Image.keep_contour_with_white_background(img.card, c)

                # Получение координат ограничивающего квадрата вокруг контура 'c'
                x, y, w, h = Image.bounding_square_around_contour(c)

                # Извлечение Области Интереса (ROI) из изображения значка с использованием координат ограничивающего квадрата
                img.roi = Image.take_out_roi(img.icon, x, y, w, h)

                if not img.roi.size == 0 and img.roi.shape[0] > 0 and img.roi.shape[1] > 0:
                    img.roi = Image.resize_image(img.roi, (400, 400))
                    img.save_image(f'test/predict{self.nr}/predict', img.roi, addition=f'_{i}')
                else:
                    white_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
                    img.save_image(f'test/predict{self.nr}/predict', white_image, addition=f'_{i}')
                i += 1

            # Предсказание
            # Получение предсказаний модели для каждого изображения в текущем объекте Image (img)

            # Получение индексов классов с наивысшей вероятностью для каждого изображения
            img.predicted_class_indices, img.predicted_probabilities = get_predictions(img.preddir, model)

            # Преобразование индексов классов в метки (названия классов)
            img.predicted_labels = indices_to_labels(img.predicted_class_indices)

            # Создание словаря, связывающего метки классов с их порядковыми номерами (индексами) в предсказаниях
            for i in range(len(img.predicted_labels)):
                img.predictions[img.predicted_labels[i]] = i

            # Для каждого изображения контура (или символа) в объекте Image
            for i in range(min(len(img.cnts_images), len(img.predicted_probabilities))):
                # Форматирование вероятности в строку с двумя знаками после запятой
                probability = '%.2f' % img.predicted_probabilities[i]

                # Создание текста с меткой класса и соответствующей вероятностью
                text = f'{img.predicted_labels[i].capitalize()}: {probability}'

                # Добавление текста к изображению контура с учетом смещения
                img.cnts_images[i] = Image.add_text(img.cnts_images[i], text, x=img.cntsx[i] - 20, y=img.cntsy[i] - 10)

                # Сохранение изображения с добавленным текстом в указанную директорию
                img.save_image(directory=f'test/predict{self.nr}', image=img.cnts_images[i], addition=f'_{i}')

            self.nr += 1

        # Сравнение комбинаций изображений
        for combo in itertools.combinations(self.images, 2):
            # Анализ предсказанных меток классов для комбинации изображений в паре (combo)

            # Создание списка предсказанных меток для каждого изображения в паре
            all_predictions = [img.predicted_labels for img in combo]

            # Нахождение общих (пересекающихся) меток классов между изображениями в паре
            common = list(set(all_predictions[0]).intersection(all_predictions[1]))

            # Инициализация списков и переменных для вычисления средних вероятностей и общих меток
            probs_means = []
            mean = 0
            common_icon = []

            # Если общая метка одна, добавляем ее в список common_icon
            if len(common) == 1:
                common_icon.append(common[0])

            # Если общих меток больше одной, вычисляем средние вероятности для каждой общей метки
            if len(common) > 1:
                for icon in common:
                    # Вычисление средней вероятности для текущей метки класса по изображениям в паре
                    mean = np.mean([img.predicted_probabilities[img.predictions[icon]] for img in combo])
                    probs_means.append(mean)

                # Нахождение индекса максимальной средней вероятности и добавление соответствующей метки в common_icon
                idx = probs_means.index(max(probs_means))
                common_icon.append(common[idx])

            # Инициализация списков для хранения изображений и имен файлов
            add_images = []
            found_name = ''

            winning = img.cnts_images[0]
            # Если найдена одна общая метка, создаем изображение с объединенными результатами
            if len(common_icon) == 1:
                # Формирование текста с результатом
                text = f'The common icon is: {common_icon[0].capitalize()}!'
                print(text)
                # Добавление изображений с выделенными символами для каждого изображения в паре
                for img in combo:
                    # Получение индекса общей метки на изображении
                    common_icon_index = img.predictions.get(common_icon[0], None)

                    # Если метка найдена, добавляем соответствующее изображение в список
                    if common_icon_index is not None and 0 <= common_icon_index < len(img.cnts_images):
                        winning = img.cnts_images[common_icon_index]
                    else:
                        print(f"Symbol {common_icon[0]} not found or not recognized.")

                    add_images.append(winning)
                    found_name += img.wo_extension


                # Объединение изображений и добавление текста
                found = Image.add_2_images(add_images[0], add_images[1])
                found = Image.add_text(found, text, x=600, y=50, thickness=4)

                found = PILImage.fromarray(found)

                found = found.resize((200, 200))
                # Convert PIL Image to PhotoImage
                found_tk = ImageTk.PhotoImage(found)


                # Configure the Label with the PhotoImage
                self.result_image_label.configure(image=found_tk)
                self.result_image_label.image = found_tk

                # Сохранение объединенного изображения
                Image.save_image_(directory='predictedG', image=found, name=f'{common_icon[0]}{found_name}')

            else:
                # Если не найдено общих меток, создаем изображение с сообщением "Не найдены иконки :("
                text = f"Didn't find icons :("

                # Добавление обработанных изображений в список
                for img in combo:
                    add_images.append(img.processed)
                    found_name += img.wo_extension

                # Объединение изображений и добавление текста
                found = Image.add_2_images(add_images[0], add_images[1])
                found = Image.add_text(found, text, x=600, y=50, thickness=4)

                # Сохранение объединенного изображения
                Image.save_image_(directory='probGUI', image=found, name=f'01_{found_name}')
                # Путь к итоговому изображению

        if remove:
            for img in self.images:
                shutil.rmtree(img.preddir)


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()
