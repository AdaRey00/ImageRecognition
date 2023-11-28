# Import necessary libraries
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from collections import defaultdict

# Load your trained model
model = load_model('C:/Users/adare/PycharmProjects/ImageRecognition/models/model.h5')

# Define the path to the test dataset
test_data_dir = 'C:/Users/adare/PycharmProjects/ImageRecognition/data/output_data/test'

# Define a dictionary to store misclassified images by class
misclassified_images_by_class = defaultdict(list)

# Iterate through the test dataset
for class_folder in os.listdir(test_data_dir):
    class_folder_path = os.path.join(test_data_dir, class_folder)

    # Iterate through images in each class folder
    for image_file in os.listdir(class_folder_path):
        image_path = os.path.join(class_folder_path, image_file)

        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize the image

        # Make a prediction using the model
        predicted_class = np.argmax(model.predict(img), axis=-1)[0]

        # Compare the predicted class with the true class (based on folder name)
        true_class = int(class_folder)  # Assuming class folder names are integers

        # If the prediction doesn't match the true class, it's misclassified
        if predicted_class != true_class:
            misclassified_images_by_class[true_class].append(image_path)

# Now, misclassified_images_by_class contains a dictionary where each key is the true class,
# and the value is a list of file paths to misclassified images for that class.

# You can further analyze these misclassified images and patterns as described in the previous response.

print(misclassified_images_by_class)