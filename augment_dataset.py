from PIL import Image
import os

def rotate_and_save_all_images(input_folder, output_folder, num_rotations):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            original_image = Image.open(input_path)

            # Convert image to RGB mode if it has an alpha channel
            if original_image.mode == 'RGBA':
                original_image = original_image.convert('RGB')

            for i in range(num_rotations):
                angle = i * (360 / num_rotations)
                rotated_image = original_image.rotate(angle, expand=True)

                output_filename = f"{os.path.splitext(filename)[0]}_rotated_{i+1}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                rotated_image.save(output_path)

# Пример использования
input_folder = "dataset_cards"
output_folder = "dataset"
num_rotations = 6

rotate_and_save_all_images(input_folder, output_folder, num_rotations)
