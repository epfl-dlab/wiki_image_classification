import os
import sys
import json
import time
import pandas as pd
import tensorflow as tf
import help_functions as hf
from matplotlib import pyplot as plt
from datetime import datetime

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


# To run this: `python TrainClassification.py 0`


# ================== HYPER-PARAMETERS ==================
i = sys.argv[1]
with open('training_configurations.json', 'r') as fp:
    config = json.load(fp)[str(i)]
print(config)
# Save outputs to log file
old_stdout = sys.stdout
os.mkdir(config['results_folder'])
log_file = open(config['results_folder'] + '/log.txt', 'w')
sys.stdout = log_file
# ======================================================


print(config)

hf.print_time(start)

# ================= LOAD DATA ================
start = time.time()
train = hf.get_flow(df_file=config['data_folder'] + '/train_df.json.bz2',
                    nr_classes=config['nr_classes'],
                    image_dimension=config['image_dimension'])

print('LOG: finished getting the first flow')
hf.print_time(start)

# If undersample:
if config['undersample']:
    start = time.time()
    y_true = hf.get_y_true(shape=(train.samples, len(train.class_indices)), classes=train.classes)
    indices_to_remove = hf.undersample(y_true, 
                                       list(train.class_indices.keys()), 
                                       0.8, 
                                       config['results_folder'])
    print('LOG: found rows to remove to balance')
    hf.print_time(start)
    start = time.time()
    train_df = pd.read_json(config['data_folder'] + '/train_df.json.bz2', compression='bz2')
    balanced_df = train_df.reset_index().drop(index=indices_to_remove)
    train = hf.get_flow(df=balanced_df,
                        nr_classes=config['nr_classes'],
                        image_dimension=config['image_dimension'])
    print('LOG: got the new balanced flow')
    hf.print_time(start)
    
start = time.time()

val_stop = hf.get_flow(df_file=config['data_folder'] + '/val_stop_df.json.bz2',
                       nr_classes=config['nr_classes'],
                       image_dimension=config['image_dimension'])
print('LOG: Got the validation flow')
hf.print_time(start)


# name_id_map = train.class_indices
# class_names = len(name_id_map)*[0]
# for k in name_id_map.keys():
#     class_names[name_id_map[k]] = k
# ======================================================



# ====================== CREATE MODEL ==================
print('LOG: creating and training model')
start = time.time()
model = hf.create_model(n_labels=config['nr_classes'], 
                        image_dimension=config['image_dimension'],
                        model_name=config['basemodel'], 
                        number_trainable_layers=config['number_trainable_layers'],
                        random_initialization=config['random_initialization'])
# ======================================================



# ===================== TRAIN MODEL ==================
# Save model in between epochs
checkpoint_path = config['results_folder'] + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 save_weights_only=True,
                                                 verbose=1)
history_callback = tf.keras.callbacks.CSVLogger(f"{config['results_folder']}/history.csv", 
                                                separator=',', 
                                                append=True)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           min_delta=0,
                                                           patience=15,
                                                           verbose=0,
                                                           mode='auto',
                                                           restore_best_weights=True)
# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))


if config['class_weights'] == True:
    pass
    # # Calculate class weights: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
    # weights = train_df[["labels", "title"]].explode("labels")\
    #         .groupby("labels").agg("count").reset_index()
    # total = weights.title.sum()
    # weights['proportion'] = weights.title.apply(lambda r: r/total)
    # weights['weight'] = weights.title.apply(lambda r: (1/r)*(total/41)) # 
    # # weights['weight'] = weights.page_title.apply(lambda r: np.log((1/r)*(total/2)))
    # weights = weights[['labels', 'proportion', 'weight']]
    # class_weight = {}
    # for l in name_id_map.keys():
    #     w = weights[weights.labels==l].weight.iloc[0]
    #     class_weight[train_generator.class_indices[l]] = w
    # history = model.fit(
    # train_generator,
    # verbose=2,
    # validation_data=validation_generator,
    # epochs=EPOCHS,
    # callbacks=[cp_callback, history_callback, early_stopping_callback],
    # class_weight=class_weight)

else:
    history = model.fit(
    train,
    verbose=2,
    validation_data=val_stop,
    epochs=config['epochs'],
    callbacks=[cp_callback, history_callback, early_stopping_callback])
# ======================================================


hf.print_time(start)


# ================= PLOT TRAINING METRICS ==============
training_metrics = pd.read_csv(config['results_folder'] + '/history.csv')

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
hf.save_img(config['results_folder'] + '/training_metrics.png')
# ======================================================