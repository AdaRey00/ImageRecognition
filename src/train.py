from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.model import create_model
from config import *

# Load the model
model = create_model()

# Prepare data generators for training, validation, and test
train_datagen = ImageDataGenerator(rescale=rescale_factor)
val_datagen = ImageDataGenerator(rescale=rescale_factor)
test_datagen = ImageDataGenerator(rescale=rescale_factor)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode)

val_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode)

# Train the model
model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)

# Save the model
model.save(model_save_path)


