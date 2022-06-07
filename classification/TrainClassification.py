import tensorflow as tf
import pandas as pd
import json
import os
import sys
from help_functions import get_top_classes
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# To run this: python TrainClassification.py 0




# ================== HYPER-PARAMETERS ==================
# config: nr_classes, labels, class_weights, basemodel, image_dimension, results_and_checkpoints_folder, data_folder
i = sys.argv[1]
with open('training_configurations.json', 'r') as fp:
    config = json.load(fp)[str(i)]
print(config)
# ======================================================




# ================= LOAD & AUGMENT DATA ================
train_df = pd.read_json(config['data_folder'] + '/train_df.json.bz2', compression='bz2')

top_classes = get_top_classes(config['nr_classes'], train_df)#['Places', 'Culture', 'History', 'Society', 'Nature', 'People', 'Politics', 'Sports', 'Objects', 'Entertainment']
print(f"Top {config['nr_classes']} classes: {top_classes}")

# Only keep rows which have either of the top classes
ids_x_labels = train_df.labels.apply(lambda classes_list: any([True for a_class in top_classes if a_class in classes_list]))
training_set_x_labels = train_df[ids_x_labels]
training_set_x_labels['labels'] = train_df['labels'].apply(lambda labels_list: [label for label in labels_list if label in top_classes])
train_df = training_set_x_labels.copy()

batch_size = 32
width, height = config['image_dimension'], config['image_dimension']
target_size = (height, width)
datagen = ImageDataGenerator(rotation_range=40, 
                             width_shift_range=0.2,
                             height_shift_range=0.2, 
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest',
                             validation_split=0.05) 

train_generator = datagen.flow_from_dataframe(
        dataframe=train_df, 
        directory='/scratch/WIT_Dataset/images/', 
        subset='training',
        color_mode='rgb',
        x_col='url', 
        y_col='labels', 
        class_mode='categorical', 
        batch_size=batch_size,
        target_size=target_size)

validation_generator = datagen.flow_from_dataframe(
        dataframe=train_df, 
        directory='/scratch/WIT_Dataset/images/', 
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
# model = create_model(config['nr_classes'], image_dimension=config['image_dimension'], model_name=config['basemodel'])
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def create_model():
    efficient_net = EfficientNetB0(include_top=False, weights='imagenet', classes=len(CLASS_LABELS),
                                           input_shape=(width, height, 3))

    efficient_net.trainable=False

    model = Sequential([
        efficient_net,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(CLASS_LABELS), activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])

    model.summary()
    return model
model = create_model()


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
print(weights)
# ======================================================






# ===================== TRAIN MODEL ==================
epochs = 15

# Save model in between epochs
checkpoint_path = config['results_and_checkpoints_folder'] + "/cp-{epoch:04d}.ckpt"
print(checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
history_callback = tf.keras.callbacks.CSVLogger(f"{config['results_and_checkpoints_folder']}/history.csv", 
                                                separator=',', 
                                                append=True)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(
  train_generator,
  verbose=1,
  validation_data=validation_generator,
  epochs=epochs,
  callbacks=[cp_callback, history_callback],
  class_weight=class_weight)
# ======================================================