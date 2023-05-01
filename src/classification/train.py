import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import help_functions as hf
from matplotlib import pyplot as plt
from hierarchical_model import HierarchicalModel
from configs import configs

# hf.setup_gpu(gpu_nr=0) # if some of the GPUs is busy, choose one (0 or 1)


# ===========================================
# =========== HYPER-PARAMETERS ==============
# ===========================================

config = configs[0]
print(config)

# Save outputs to log file
old_stdout = sys.stdout
os.mkdir(config['results_folder'])
log_file = open(config['results_folder'] + '/log.txt', 'w')
sys.stdout = log_file


# ============================================
# ================= LOAD DATA ================
# ============================================
train, train_df = hf.get_flow(df_file=config['data_folder'] + '/train_df.json.bz2',
                              batch_size=config['batch_size'],
                              image_dimension=config['image_dimension'])
print('LOG: finished getting training flow')

y_true = hf.get_y_true(shape=(train.samples, len(train.class_indices)), classes=train.classes)

val_stop, _ = hf.get_flow(df_file=config['data_folder'] + '/val_df.json.bz2',
                          batch_size=config['batch_size'],
                          image_dimension=config['image_dimension'])
print('LOG: Got the validation flow')



# ============================================
# ============= CREATE MODEL =================
# ============================================
print('LOG: creating and training model')
start = time.time()
if config['hierarchical']:
    model = HierarchicalModel(nr_labels=len(train.class_indices), image_dimension=config['image_dimension'])
else:
    model = hf.create_model(n_labels=len(train.class_indices), 
                            image_dimension=config['image_dimension'],
                            number_trainable_layers=config['number_trainable_layers'],
                            random_initialization=config['random_initialization'],
                            loss=config['loss_function'],
                            y_true=y_true)



# ============================================
# ================ TRAIN MODEL ===============
# ============================================
# Save model in-between epochs
checkpoint_path = config['results_folder'] + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

if config['monitor'] == 'pr_auc':
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_best_only=False,
                                                    monitor='val_pr_auc', 
                                                    mode='max',
                                                    save_weights_only=True,
                                                    verbose=1)
else:
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_best_only=False,
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
        callbacks=[cp_callback, history_callback],
        use_multiprocessing=True,
        workers=10)
hf.print_time(start)



# =========================================
# ========= PLOT TRAINING METRICS =========
# =========================================

training_metrics = pd.read_csv(config['results_folder'] + '/history.csv')

epochs = training_metrics.shape[0]

plt.figure(figsize=(25,10))

_ = plt.subplot(1, 5, 1)
plt.plot(range(config['epochs']), training_metrics.loss.values, label='Training loss')
plt.plot(range(config['epochs']), training_metrics.val_loss.values, label='Validation loss')
plt.xlabel('Epochs')
plt.title('Loss')
plt.xticks(np.arange(0, config['epochs'], step=2), np.arange(1, config['epochs'], step=2))
plt.legend(['Train', 'Validation'])

_ = plt.subplot(1, 5, 2)
plt.plot(range(config['epochs']), training_metrics.recall.values, label='Training recall')
plt.plot(range(config['epochs']), training_metrics.val_recall.values, label='Validation recall')
plt.title('Recall')
plt.xlabel('Epochs')
plt.xticks(np.arange(0, config['epochs'], step=2), np.arange(1, config['epochs'], step=2))
plt.legend(['Train', 'Validation'])

_ = plt.subplot(1, 5, 3)
plt.plot(range(config['epochs']), training_metrics.precision.values, label='Training precision')
plt.plot(range(config['epochs']), training_metrics.val_precision.values, label='Validation precision')
plt.title('Precision')
plt.xlabel('Epochs')
plt.xticks(np.arange(0, config['epochs'], step=2), np.arange(1, config['epochs'], step=2))
plt.legend(['Train', 'Validation'])

_ = plt.subplot(1, 5, 4)
plt.plot(range(config['epochs']), training_metrics.pr_auc.values, label='Training PR_AUC')
plt.plot(range(config['epochs']), training_metrics.val_pr_auc.values, label='Validation PR_AUC')
plt.title('PR AUC')
plt.xlabel('Epochs')
plt.xticks(np.arange(0, config['epochs'], step=2), np.arange(1, config['epochs'], step=2))
plt.legend(['Train', 'Validation'])

_ = plt.subplot(1, 5, 5)
plt.plot(range(config['epochs']), training_metrics.binary_accuracy.values, label='Training binary acc')
plt.plot(range(config['epochs']), training_metrics.val_binary_accuracy.values, label='Validation binary acc')
plt.title('Binary accuracy')
plt.xlabel('Epochs')
plt.xticks(np.arange(0, config['epochs'], step=2), np.arange(1, config['epochs'], step=2))
plt.legend(['Train', 'Validation'])

hf.save_img(config['results_folder'] + '/training_metrics.png')