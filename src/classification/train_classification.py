﻿import os
import sys
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import help_functions as hf
from matplotlib import pyplot as plt
from hierarchical_model import HierarchicalModel

# hf.setup_gpu(gpu_nr=0)

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


# ================= LOAD DATA ================
start = time.time()
train, train_df = hf.get_flow(df_file=config['data_folder'] + '/train_df.json.bz2',
                              batch_size=config['batch_size'],
                              image_dimension=config['image_dimension'])

print('LOG: finished getting training flow')
hf.print_time(start)

y_true = hf.get_y_true(shape=(train.samples, len(train.class_indices)), classes=train.classes)

if config['undersample']:
    indices_to_remove = hf.undersample(y_true, 
                                       list(train.class_indices.keys()), 
                                       0.8, 
                                       config['results_folder'])
    balanced_df = train_df.reset_index().drop(index=indices_to_remove) 
    train, _ = hf.get_flow(df=balanced_df,
                           batch_size=config['batch_size'],
                           image_dimension=config['image_dimension'])
    print('LOG: got the new undersampled flow')
elif config['oversample']:
    start = time.time()
    print('LOG: oversampling....')
    duplicate_indices_dict = hf.oversample(y_true, 
                                           list(train.class_indices.keys()), 
                                           0.2, 
                                           config['results_folder'])
    hf.print_time(start)
    print('LOG: found indices to duplicate...')
    start = time.time()
    idx_to_duplicate_from = y_true.shape[0] # the oversampled images will be added after this index
    for index_to_duplicate in duplicate_indices_dict:
        times_to_duplicate = duplicate_indices_dict[index_to_duplicate]
        train_df = pd.concat([train_df, pd.DataFrame([train_df.iloc[index_to_duplicate]] * times_to_duplicate)], axis=0, ignore_index=True)
    train_df = train_df.reset_index() 
    # Take number of rows divisible by batch_size otherwise training will fail
    train_df = train_df.iloc[0:config['batch_size']*(train_df.shape[0] // config['batch_size'])]
    print('LOG: starting to get oversampled new flow with number of rows divisible by batch size...')
    train, _ = hf.get_flow(df=train_df,
                           batch_size=config['batch_size'],
                           image_dimension=config['image_dimension'])
    if config['augment']:
        train_augment = hf.augment(train, config['batch_size'], config['image_dimension'], idx_to_duplicate_from)
        print('LOG: got augmented dataset')

    print('LOG: got the new oversampled flow')
    hf.print_time(start)

start = time.time()

val_stop, _ = hf.get_flow(df_file=config['data_folder'] + '/val_df.json.bz2',
                          batch_size=config['batch_size'],
                          image_dimension=config['image_dimension'])

print('LOG: Got the validation flow')
hf.print_time(start)
# ======================================================



# ====================== CREATE MODEL ==================
print('LOG: creating and training model')
start = time.time()
if config['hierarchical']:
    print('LOG: Getting hierarchical model...')
    model = HierarchicalModel(nr_labels=len(train.class_indices), image_dimension=config['image_dimension'])
else:
    model = hf.create_model(n_labels=len(train.class_indices), 
                            image_dimension=config['image_dimension'],
                            model_name=config['basemodel'], 
                            number_trainable_layers=config['number_trainable_layers'],
                            random_initialization=config['random_initialization'],
                            y_true=y_true)
# ======================================================



# ===================== TRAIN MODEL ==================
# Save model in-between epochs
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

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

if config['class_weights'] == True:
    class_weights = hf.compute_class_weights(y_true)
    history = model.fit(
        train,
        verbose=2,
        validation_data=val_stop,
        epochs=config['epochs'],
        callbacks=[cp_callback, history_callback],
        class_weight=class_weights)
elif config['augment'] == True:
    # steps_per_epoch = (train.samples // config['batch_size'])
    # validation_steps = (val_stop.samples // config['batch_size'])
    history = model.fit(
        train_augment, 
        # steps_per_epoch=steps_per_epoch,
        verbose=2,
        validation_data=val_stop,
        # validation_steps=validation_steps,
        epochs=config['epochs'],
        callbacks=[cp_callback, history_callback])
elif config['hierarchical']:
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss='binary_crossentropy', 
                  metrics=[
                           tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.AUC(num_thresholds=50, curve='PR', name='pr_auc', multi_label=True),
                        ])
    history = model.fit(train, 
                        verbose=2,
                        epochs=config['epochs'], 
                        validation_data=val_stop,
                        callbacks=[cp_callback, history_callback]) 
else:
    history = model.fit(
        train,
        verbose=2,
        validation_data=val_stop,
        epochs=config['epochs'],
        callbacks=[cp_callback, history_callback])
hf.print_time(start)
# ======================================================




# ================= PLOT TRAINING METRICS ==============
training_metrics = pd.read_csv(config['results_folder'] + '/history.csv')

epochs = training_metrics.shape[0]

_ = plt.subplot(1, 5, 1)
plt.plot(range(config['epochs']), training_metrics.loss.values, label='Training loss')
plt.plot(range(config['epochs']), training_metrics.val_loss.values, label='Validation loss')
plt.xlabel('Epochs')
plt.title('Loss')
plt.xticks(np.arange(0, 20, step=2), np.arange(1, 20, step=2))
plt.legend(['Train', 'Validation'])

_ = plt.subplot(1, 5, 2)
plt.plot(range(config['epochs']), training_metrics.recall.values, label='Training recall')
plt.plot(range(config['epochs']), training_metrics.val_recall.values, label='Validation recall')
plt.title('Recall')
plt.xlabel('Epochs')
plt.xticks(np.arange(0, 20, step=2), np.arange(1, 20, step=2))
plt.legend(['Train', 'Validation'])

_ = plt.subplot(1, 5, 3)
plt.plot(range(config['epochs']), training_metrics.precision.values, label='Training precision')
plt.plot(range(config['epochs']), training_metrics.val_precision.values, label='Validation precision')
plt.title('Precision')
plt.xlabel('Epochs')
plt.xticks(np.arange(0, 20, step=2), np.arange(1, 20, step=2))
plt.legend(['Train', 'Validation'])

_ = plt.subplot(1, 5, 4)
plt.plot(range(config['epochs']), training_metrics.pr_auc.values, label='Training PR_AUC')
plt.plot(range(config['epochs']), training_metrics.val_pr_auc.values, label='Validation PR_AUC')
plt.title('PR AUC')
plt.xlabel('Epochs')
plt.xticks(np.arange(0, 20, step=2), np.arange(1, 20, step=2))
plt.legend(['Train', 'Validation'])

_ = plt.subplot(1, 5, 5)
plt.plot(range(config['epochs']), training_metrics.binary_accuracy.values, label='Training binary acc')
plt.plot(range(config['epochs']), training_metrics.val_binary_accuracy.values, label='Validation binary acc')
plt.title('Binary accuracy')
plt.xlabel('Epochs')
plt.xticks(np.arange(0, 20, step=2), np.arange(1, 20, step=2))
plt.legend(['Train', 'Validation'])

hf.save_img(config['results_folder'] + '/training_metrics.png')


# ======================================================