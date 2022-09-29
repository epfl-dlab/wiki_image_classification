import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import sys
from help_functions import get_top_classes
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from focal_loss import BinaryFocalLoss

# Sharing of GPU resources
# tf.config.gpu.set_per_process_memory_growth(True) # TODO didn't work, got error: AttributeError: module 'tensorflow._api.v2.config' has no attribute 'gpu'
# tf.config.threading.set_intra_op_parallelism_threads(10) 
# tf.config.threading.set_inter_op_parallelism_threads(10) 

start = time.time()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# To run this: `python TrainClassification.py 0`


# ================== HYPER-PARAMETERS ==================
LOSS_FUNCTION = 'binary_crossentropy'
# LOSS_FUNCTION = 'binary_focal_crossentropy'
BATCH_SIZE = 512
EPOCHS = 20

# config: nr_classes, labels, class_weights, basemodel, image_dimension, results_and_checkpoints_folder, data_folder
i = sys.argv[1]
with open('training_configurations.json', 'r') as fp:
    config = json.load(fp)[str(i)]
print(config)
# Save outputs to log file
old_stdout = sys.stdout
os.mkdir(config['results_and_checkpoints_folder'])
log_file = open(config['results_and_checkpoints_folder'] + '/log.txt', 'w')
sys.stdout = log_file
# ======================================================


print(f'\nBATCH SIZE: {BATCH_SIZE}')
print(f'\LOSS_FUNCTION SIZE: {LOSS_FUNCTION}')
print(f'\EPOCHS SIZE: {EPOCHS}\n')


# ================= LOAD & AUGMENT DATA ================
train_df = pd.read_json(config['data_folder'] + '/train_df.json.bz2', compression='bz2')
top_classes = get_top_classes(config['nr_classes'], train_df)#['Places', 'Culture', 'History', 'Society', 'Nature', 'People', 'Politics', 'Sports', 'Objects', 'Entertainment']
print(f"Top {config['nr_classes']} classes: {top_classes}")

# Only keep rows which have either of the top classes
ids_x_labels = train_df.labels.apply(lambda classes_list: any([True for a_class in top_classes if a_class in classes_list]))
training_set_x_labels = train_df[ids_x_labels]
training_set_x_labels['labels'] = train_df['labels'].apply(lambda labels_list: [label for label in labels_list if label in top_classes])
train_df = training_set_x_labels.copy()

width, height = config['image_dimension'], config['image_dimension']
target_size = (height, width)
if config['augment'] == True:
    datagen = ImageDataGenerator(rotation_range=40, 
                                width_shift_range=0.2,
                                height_shift_range=0.2, 
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest',
                                validation_split=0.05) 
else:
    datagen = ImageDataGenerator(validation_split=0.05)

train_generator = datagen.flow_from_dataframe(
        dataframe=train_df, 
        directory='/scratch/WIT_Dataset/images/', 
        seed=7,
        subset='training',
        color_mode='rgb',
        x_col='url', 
        y_col='labels', 
        class_mode='categorical', 
        batch_size=BATCH_SIZE,
        target_size=target_size)

validation_generator = datagen.flow_from_dataframe(
        dataframe=train_df, 
        directory='/scratch/WIT_Dataset/images/', 
        seed=7,
        subset='validation',
        color_mode='rgb',
        x_col='url', 
        y_col='labels', 
        class_mode='categorical', 
        target_size=target_size)

name_id_map = train_generator.class_indices
class_names = len(name_id_map)*[0]
for k in name_id_map.keys():
    class_names[name_id_map[k]] = k

class_indices = train_generator.class_indices
CLASS_LABELS = list(class_indices.keys())
# ======================================================


# ====================== CREATE MODEL ==================
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def create_model(name):
    if name == 'EfficientNetB0':
        efficient_net = EfficientNetB0(include_top=False, weights='imagenet', classes=len(CLASS_LABELS),
                                           input_shape=(width, height, 3))
    elif name == 'EfficientNetB1':
        efficient_net = EfficientNetB1(include_top=False, weights='imagenet', classes=len(CLASS_LABELS),
                                           input_shape=(width, height, 3))
    elif name == 'EfficientNetB2':
        efficient_net = EfficientNetB2(include_top=False, weights='imagenet', classes=len(CLASS_LABELS),
                                           input_shape=(width, height, 3))

    print(f'\nNumber of layers in basemodel: {len(efficient_net.layers)}')
    # Fine tune from this layer onwards
    # fine_tune_at = round((1 - config['percent_trainable_layers']) * len(efficient_net.layers))
    number_trainable_layers = config['number_trainable_layers']
    fine_tune_at = len(efficient_net.layers) - number_trainable_layers
 
    print(f'Number of trainable layers: {len(efficient_net.layers) - fine_tune_at}\n')
    for layer in efficient_net.layers[:fine_tune_at]:
        layer.trainable = False

    model = Sequential([
        efficient_net,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(CLASS_LABELS), activation='sigmoid')
    ])

    # Losses:
    #   - binary cross-entropy: https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
    #   - binary focal cross-entropy: https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryFocalCrossentropy

    if LOSS_FUNCTION == 'binary_crossentropy':
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'categorical_accuracy'])
    # Binary Focal Cross Entropy
    elif LOSS_FUNCTION == 'binary_focal_crossentropy':
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=BinaryFocalLoss(gamma=2, from_logits=False),
                    metrics=['accuracy', 'categorical_accuracy'])

    model.summary()
    return model
model = create_model(name=config['basemodel'])
# ======================================================



# ===================== TRAIN MODEL ==================
# Save model in between epochs
checkpoint_path = config['results_and_checkpoints_folder'] + "/cp-{epoch:04d}.ckpt"
print(checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 save_weights_only=True,
                                                 verbose=1)
history_callback = tf.keras.callbacks.CSVLogger(f"{config['results_and_checkpoints_folder']}/history.csv", 
                                                separator=',', 
                                                append=True)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           min_delta=0,
                                                           patience=3,
                                                           verbose=0,
                                                           mode='auto',
                                                           restore_best_weights=True)
# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

if config['class_weights'] == True:
    # Calculate class weights: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
    weights = train_df[["labels", "title"]].explode("labels")\
            .groupby("labels").agg("count").reset_index()
    total = weights.title.sum()
    weights['proportion'] = weights.title.apply(lambda r: r/total)
    weights['weight'] = weights.title.apply(lambda r: (1/r)*(total/41)) # 
    # weights['weight'] = weights.page_title.apply(lambda r: np.log((1/r)*(total/2)))
    weights = weights[['labels', 'proportion', 'weight']]
    class_weight={}
    for l in name_id_map.keys():
        w = weights[weights.labels==l].weight.iloc[0]
        class_weight[train_generator.class_indices[l]] = w
    history = model.fit(
    train_generator,
    verbose=2,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[cp_callback, history_callback, early_stopping_callback],
    class_weight=class_weight)

else:
    history = model.fit(
    train_generator,
    verbose=2,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[cp_callback, history_callback, early_stopping_callback])
# ======================================================


end = time.time()
total_time_in_hours = round((end - start) / 3600)
print(f'\nTraining time: {total_time_in_hours} hours\n')





# ===== FIND OPTIMAL THRESHOLDS (using val set) ========
thresholds = np.linspace(0, 1, 0.001)
val_y = model.predict


# ======================================================




# ================= PLOT TRAINING METRICS ==============
# Plot training metrics: loss & accuracy
training_metrics = pd.read_csv(config['results_and_checkpoints_folder'] + '/history.csv')

epochs = training_metrics.shape[0]

acc = training_metrics.accuracy.values
loss = training_metrics.loss.values

val_acc = training_metrics.val_accuracy.values
val_loss = training_metrics.val_loss.values

_ = plt.figure(figsize=(12, 4))
_ = plt.subplot(1, 3, 1)
_ = plt.plot(range(epochs), acc, label='Training Accuracy')
_ = plt.plot(range(epochs), val_acc, label='Validation Accuracy')
_ = plt.legend(loc='lower right')
_ = plt.title('Training and Validation Accuracy')

_ = plt.subplot(1, 3, 2)
_ = plt.plot(range(epochs), loss, label='Training Loss')
_ = plt.legend(loc='upper right')
_ = plt.title('Training Loss')

_ = plt.subplot(1, 3, 3)
_ = plt.plot(range(epochs), val_loss, label='Validation Loss', color='orange')
_ = plt.legend(loc='upper right')
_ = plt.title('Validation Loss')
_ = plt.savefig(config['results_and_checkpoints_folder'] + '/training_metrics.png')
# ======================================================