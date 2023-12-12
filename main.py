from computervision.opencvutils import Image
from computervision.utils import *
from computervision.kerasutils import *
from computervision.imageioutils import *

import time
from os.path import join
import shutil
import itertools

start = time.perf_counter()

gif = True
remove = False

number_combinations = 1428
# Загрузка модели CNN
model = load_cnn_model()

#создание тестового генератора
test_dir = 'symbols/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(400, 400), batch_size=20, class_mode='categorical')

#датасет
directory = 'dataset_cards'
files = list_files_in_dir(directory)

# Создание объектов изображений для обработки
images = [Image(path_to_img=join(directory, files[i])) for i in range(len(files))]
nr = 0
for img in images:
    # Обработка изображений для предсказания, добавление контраста, изменение размера, размытие
    img.processed = img.contrast_resized_blurred()
    # Создание директории для предсказаний
    img.preddir = create_prediction_dirs(nr)
    # Нахождение самого большого контура: это карта
    cnt = Image.grab_contours_by_area(img.processed)[0]
    img.card = Image.keep_contour_with_white_background(img.processed, cnt)
    # Нахождение 10 самых больших контуров, как минимум, для восьми иконок на карте
    img.cnts = Image.grab_contours_by_area(img.card, reverse=True, threshold=190, area=900)

    i = 0
    for c in img.cnts:
        x, y, w, h = Image.get_rect_coordinates_around_contour(c)
        img.cntsx.append(x)
        img.cntsy.append(y)
        img.drawncontour = Image.draw_contour(img.processed.copy(), c)
        img.cnts_images.append(img.drawncontour)
        img.icon = Image.keep_contour_with_white_background(img.card, c)
        x, y, w, h = Image.bounding_square_around_contour(c)
        img.roi = Image.take_out_roi(img.icon, x, y, w, h)

        if not img.roi.size == 0 and img.roi.shape[0] > 0 and img.roi.shape[1] > 0:
            img.roi = Image.resize_image(img.roi, (400, 400))
            img.save_image(f'test/predict{nr}/predict', img.roi, addition=f'_{i}')
        else:
            white_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
            img.save_image(f'test/predict{nr}/predict', white_image, addition=f'_{i}')
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
        img.save_image(directory=f'test/predict{nr}', image=img.cnts_images[i], addition=f'_{i}')

    if gif:
        create_gif(img.preddir, f'{img.wo_extension}')

    nr += 1

# Сравнение комбинаций изображений
for combo in itertools.combinations(images, 2):
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

    # Если найдена одна общая метка, создаем изображение с объединенными результатами
    if len(common_icon) == 1:
        # Формирование текста с результатом
        text = f'The common icon is: {common_icon[0].capitalize()}!'

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

        # Сохранение объединенного изображения
        Image.save_image_(directory='predicted', image=found, name=f'{common_icon[0]}{found_name}')
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
        Image.save_image_(directory='predicted', image=found, name=f'01_{found_name}')

if remove:
    for img in images:
        shutil.rmtree(img.preddir)

end = time.perf_counter()
totaltime = end - start


print("Среднее время на поиск одной комбинаций (из двух карточек):", totaltime/number_combinations, 'секунд!')
print(f'Компьютер собрал все комбинации за {totaltime} сек!А ты?')
