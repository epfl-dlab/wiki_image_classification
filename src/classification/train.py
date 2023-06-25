import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import help_functions as hf
from matplotlib import pyplot as plt
from hierarchical_model import HierarchicalModel
from configs import configs
import csv
import urllib
LARGE_ENOUGH_NUMBER = 100
start_all = time.time()

hf.setup_gpu(gpu_nr=0) # if some of the GPUs is busy, choose one (0 or 1)

resume_training = False

# ===========================================
# =========== HYPER-PARAMETERS ==============
# ===========================================

config = configs[0]
print(config)

# Save outputs to log file
# old_stdout = sys.stdout
# os.mkdir(config['results_folder'])
# log_file = open(config['results_folder'] + '/log.txt', 'w')
# sys.stdout = log_file


# ============================================
# ================= LOAD DATA ================
# ============================================
train, _ = hf.get_flow(df_file=config['data_folder'] + '/train_df.json.bz2',
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
                            random_initialization=config['random_initialization'],
                            loss=config['loss_function'],
                            y_true=y_true)

# if resume_training:
#     latest = tf.train.latest_checkpoint(config['results_folder'])
#     print(latest)
#     model.load_weights(latest)


# ============================================
# ================ TRAIN MODEL ===============
# ============================================
# Save model in-between epochs
checkpoint_path = config['results_folder'] + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# if config['monitor'] == 'pr_auc':
#     cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                     save_best_only=True,
#                                                     monitor='val_pr_auc', 
#                                                     mode='max',
#                                                     save_weights_only=True,
#                                                     verbose=1)
# else:
#     cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                     save_best_only=True,
#                                                     monitor='val_loss',
#                                                     save_weights_only=True,
#                                                     verbose=1)
    
history_callback = tf.keras.callbacks.CSVLogger(f"{config['results_folder']}/history.csv", 
                                                separator=',', 
                                                append=True)

# Load human-labeled set
human_df = pd.read_parquet('../../data/evaluation/annotated_validation.parquet')
human_df['labels'] = human_df.apply(lambda x: list(x.labels), axis=1) # otherwise the labels column will be a list of lists
human_df['url'] = human_df.apply(lambda x: '/scratch/WIT_Dataset/images/' + x.url.split('/wikipedia/commons/')[1], axis=1)
human_df['url'] = human_df['url'].apply(lambda encoded_filename : urllib.parse.unquote(encoded_filename).encode().decode('unicode-escape'))
human, _ = hf.get_flow(df=human_df, batch_size=config['batch_size'], image_dimension=config['image_dimension'])

class EvaluateCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataflow, history_csv_path):
        super(EvaluateCallback, self).__init__()
        self.dataflow = dataflow
        self.history_csv_path = history_csv_path

    def on_epoch_end(self, epoch, logs=None):
        evaluation_results = self.model.evaluate(self.dataflow)
        loss = evaluation_results[0]
        metrics_values = evaluation_results[1:]
        metrics_names = self.model.metrics_names[1:]

        print(f'\nEpoch {epoch}: loss {loss}')
        for name, value in zip(metrics_names, metrics_values):
            print(f'{name}: {value}')

        with open(self.history_csv_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch] + [loss] + metrics_values)

evaluate_callback = EvaluateCallback(dataflow=human, history_csv_path=f"{config['results_folder']}/history_human.csv")

# Save the weights using the `checkpoint_path` format
# if resume_training:
#     # TODO: set model.save_weights(checkpoint_path.format(epoch=EPOCH_TO_START_TRAINING_FROM))
#     pass
model.save_weights(checkpoint_path.format(epoch=0))

if config['class_weights'] == True:
    class_weights = hf.compute_class_weights(y_true)
    history = model.fit(
        train,
        verbose=2,
        validation_data=val_stop,
        epochs=config['epochs'],
        callbacks=[history_callback, evaluate_callback],
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
                        callbacks=[history_callback, evaluate_callback]) 
else:
    print('\n\nStarting training...')
    history = model.fit(
        train,
        verbose=2,
        validation_data=val_stop,
        epochs=config['epochs'],
        # initial_epoch=50, # TODO: if resume_training is True, write here what the last epoch was
        callbacks=[history_callback, evaluate_callback],
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
plt.plot(range(config['epochs']), training_metrics.precision.values, label='Training precision')
plt.plot(range(config['epochs']), training_metrics.val_precision.values, label='Validation precision')
plt.title('Precision')
plt.xlabel('Epochs')
plt.xticks(np.arange(0, config['epochs'], step=2), np.arange(1, config['epochs'], step=2))
plt.legend(['Train', 'Validation'])

_ = plt.subplot(1, 5, 3)
plt.plot(range(config['epochs']), training_metrics.recall.values, label='Training recall')
plt.plot(range(config['epochs']), training_metrics.val_recall.values, label='Validation recall')
plt.title('Recall')
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

print('\n\nTotal training time:')
hf.print_time(start_all)