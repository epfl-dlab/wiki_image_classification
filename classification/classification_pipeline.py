import tensorflow as tf
import pandas as pd
import os
from help_functions import create_model, compute_class_weights, get_y_true
from PIL import PngImagePlugin  

print(tf.__version__)
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# 0. Hyperparameters
EPOCHS = 15
IMAGE_DIMENSION = 64
MINIMAL_NR_IMAGES = 1_000
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2) # to avoid corrupted .png images


# 1. Load train and test dataframes, create augmentations
train_df = pd.read_json('data/splitted_dfs_500k_20220602/train_df.json.bz2', compression='bz2')
test_df = pd.read_json('data/splitted_dfs_500k_20220602/test_df.json.bz2', compression='bz2')

# Data generator for training and validation sets, performs data augmentations
train_generator = ImageDataGenerator(validation_split=0.05, 
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest') 

print('\n----------- Train images -----------')
train = train_generator.flow_from_dataframe(dataframe=train_df, 
                                            directory='/scratch/WIT_Dataset/images', 
                                            x_col='url', 
                                            y_col='labels', 
                                            class_mode='categorical',
                                            subset='training',
                                            batch_size=32,
                                            validate_filenames=True,
                                            target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))

print('\n----------- Validation images -----------')          
val = train_generator.flow_from_dataframe(dataframe=train_df, 
                                          directory='/scratch/WIT_Dataset/images', 
                                          x_col='url', 
                                          y_col='labels', 
                                          class_mode='categorical',
                                          subset='validation',
                                          validate_filenames=True,
                                          target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))

# Data generator for test set
# test_generator = ImageDataGenerator() 
# print('\n----------- Test images -----------')          
# test = test_generator.flow_from_dataframe(dataframe=test_df,
#                                           directory='/scratch/WIT_Dataset/images',
#                                           x_col='url', 
#                                           y_col='labels', 
#                                           batch_size=32,
#                                           class_mode='categorical',
#                                           validate_filenames=True,
#                                           target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))

N_LABELS = len(train.class_indices)

model = create_model(model_name='EfficientNetB0', n_labels=N_LABELS, image_dimension=IMAGE_DIMENSION)
y_true = get_y_true(train.samples, train.class_indices, train.classes)
class_weights = compute_class_weights(y_true)
print(class_weights)

# Save the weights using the `checkpoint_path` format
# https://www.youtube.com/watch?v=HxtBIwfy0kM
# https://www.tensorflow.org/tutorials/keras/save_and_load#checkpoint_callback_usage
checkpoint_path = "checkpoints/naive_26_labels_weights_20220602/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                 save_weights_only=True, 
                                                 save_freq='epoch', # save after each epoch
                                                 verbose=1)
# https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
history_callback = tf.keras.callbacks.CSVLogger('checkpoints/naive_26_labels_weights_20220602/history.csv', 
                                                separator=',', 
                                                append=True)

model.save(checkpoint_path.format(epoch=0))

history = model.fit(train, 
                    epochs=EPOCHS, 
                    validation_data=val,
                    verbose=1,
                    callbacks=[cp_callback, history_callback],
                    class_weight=class_weights,
                   )