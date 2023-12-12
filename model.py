# # imports
# from keras import layers
# from keras import models
# from keras import optimizers
# from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# # layers, activation layer with 57 nodes (one for every symbol)
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(57, activation='softmax'))
# model.compile(loss='categorical_crossentropy',       optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
# # specify the directories
#
# train_dir = 'symbols/train'
# validation_dir = 'symbols/validation'
# test_dir = 'symbols/test'
# # data augmentation with ImageDataGenerator from Keras (only train)
# train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, vertical_flip=True)
# test_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(train_dir, target_size=(400,400), batch_size=20, class_mode='categorical')
# validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(400,400), batch_size=20, class_mode='categorical')
#
# history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)
# # don't forget to save your model!
# model.save('models/model.h5')

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the saved model
model = load_model('models/model.h5')

# Load the image you want to make predictions on
image_path = 'dataset_cards/1.jpg'  # Replace with the path to your image
img = image.load_img(image_path, target_size=(400, 400))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale to match the training data preprocessing

# Make predictions
predictions = model.predict(img_array)

# Assuming you have a dictionary that maps class indices to class names
class_indices = {0: 'class_0', 1: 'class_1', 2: 'class_2'}  # Update with your actual mapping

# Get information about predictions
predicted_labels = np.argmax(predictions, axis=1)
predicted_probabilities = np.max(predictions, axis=1)

# Display the image along with bounding boxes and class names
plt.imshow(img)

ax = plt.gca()

for label, probability in zip(predicted_labels, predicted_probabilities):
    class_name = class_indices.get(label, 'Unknown')

    # Draw a bounding box
    rect = patches.Rectangle((0, 0), 400, 400, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Annotate with class name and probability
    plt.text(10, 10, f'{class_name}: {probability:.2f}', color='r', backgroundcolor='w')

plt.axis('off')
plt.show()


