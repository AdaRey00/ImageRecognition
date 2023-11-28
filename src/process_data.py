import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_folder = "C:/Users/adare/PycharmProjects/ImageRecognition/data/image"
output_data = "C:/Users/adare/PycharmProjects/ImageRecognition/data/output_data"

splitfolders.ratio(input_folder, output=output_data, seed=42, ratio=(.7, .15, .15))


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    # ... other augmentation options ...
)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/adare/PycharmProjects/ImageRecognition/data/output_data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_val_datagen.flow_from_directory(
    'C:/Users/adare/PycharmProjects/ImageRecognition/data/output_data/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_val_datagen.flow_from_directory(
    'C:/Users/adare/PycharmProjects/ImageRecognition/data/output_data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)




