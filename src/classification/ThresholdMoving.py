import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from help_functions import create_model, get_top_classes
import time
import sys

tf.config.threading.set_intra_op_parallelism_threads(10) 
tf.config.threading.set_inter_op_parallelism_threads(10) 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)


start = time.time()

# Constants
DATA_FOLDER = 'data/split_dataframes_heuristic_labels_20220914'
RESULTS_FOLDER = 'thesis_experiments/imbalance_techniques_32bs_15epochs/0_algo_Falseaugment_20classes_Falseweights_EfficientNetB2'
NR_LABELS = 20
BATCH_SIZE = 32
TARGET_SIZE = (64, 64)

old_stdout = sys.stdout
log_file = open(RESULTS_FOLDER + '/threshold_moving.txt', 'w')
sys.stdout = log_file

# Create model and load weights
model = create_model(n_labels=NR_LABELS, image_dimension=TARGET_SIZE[0], model_name='EfficientNetB2')
latest = tf.train.latest_checkpoint(RESULTS_FOLDER + '/checkpoints')
print(latest)
model.load_weights(latest)


# Load training data
train_df = pd.read_json(DATA_FOLDER + '/train_df.json.bz2', compression='bz2')
top_classes = get_top_classes(NR_LABELS, train_df)#['Places', 'Culture', 'History', 'Society', 'Nature', 'People', 'Politics', 'Sports', 'Objects', 'Entertainment']
print(f"Top {NR_LABELS} labels: {top_classes}")
# Only keep rows which have either of the top classes
ids_x_labels = train_df.labels.apply(lambda classes_list: any([True for a_class in top_classes if a_class in classes_list]))
training_set_x_labels = train_df[ids_x_labels]
training_set_x_labels['labels'] = train_df['labels'].apply(lambda labels_list: [label for label in labels_list if label in top_classes])
train_df = training_set_x_labels.copy()

datagen = ImageDataGenerator(validation_split=0.05)
train = datagen.flow_from_dataframe(
        dataframe=train_df, 
        directory='/scratch/WIT_Dataset/images/', 
        seed=7,
        subset='training',
        color_mode='rgb',
        x_col='url', 
        y_col='labels', 
        class_mode='categorical', 
        batch_size=BATCH_SIZE,
        target_size=TARGET_SIZE)
end = time.time()
total_time_in_hours = round((end - start) / 3600)
print(f'\Time to load data and model weights: {total_time_in_hours} hours\n')


# Predict on training data and save it
start = time.time()
prediction_probs = model.predict(train, verbose=2)
print(prediction_probs.shape)
np.save(file=RESULTS_FOLDER + '/prediction_probs', arr=prediction_probs)
end = time.time()
total_time_in_hours = round((end - start) / 3600)
print(f'\Time to predict on training data: {total_time_in_hours} hours\n')

start = time.time()

# Save the y_true matrix
y_true = np.zeros(prediction_probs.shape)
for row_idx, row in enumerate(train.classes):
    for idx in row:
        y_true[row_idx, idx] = 1
np.save(file=RESULTS_FOLDER + '/y_true_train', arr=y_true)

end = time.time()
total_time_in_hours = round((end - start) / 3600)
print(f'\Time to create y_true matrix: {total_time_in_hours} hours\n')