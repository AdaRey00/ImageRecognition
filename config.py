# Configuration for image data preprocessing
image_size = (244, 244)
batch_size = 32
rescale_factor = 1./255

# Configuration for model training
num_epochs = 10
class_mode = 'categorical'

# Paths configuration
train_data_dir = '../data/output_data/train'
validation_data_dir = '../data/output_data/val'
test_data_dir = '../data/output_data/test'
model_save_path = '../models/model.keras'
