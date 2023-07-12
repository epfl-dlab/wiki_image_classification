import os
import time
import sys
import pandas as pd
import tensorflow as tf
import help_functions as hf
from configs import configs
import urllib
from help_functions import EvaluateCallback
start_all = time.time()


hf.setup_gpu(gpu_nr=0) # if some of the GPUs is busy, choose one (0 or 1)

# ===========================================
# =========== HYPER-PARAMETERS ==============
# ===========================================

# Check if an argument is provided
if len(sys.argv) > 1:
    # Access the first command-line argument (after the script name)
    argument = int(sys.argv[1])
    print("Argument:", argument)
else:
    raise('Provide a training configuration')

config = configs[argument]
print(config)
os.mkdir(config['results_folder'])

# Save outputs to log file
old_stdout = sys.stdout
log_file = open(config['results_folder'] + '/log.txt', 'w')
sys.stdout = log_file


# ============================================
# ================= LOAD DATA ================
# ============================================
train, _ = hf.get_flow(df_file=config['data_folder'] + '/train_df.json.bz2',
                              batch_size=config['batch_size'],
                              image_dimension=config['image_dimension'])
print('LOG: finished getting training flow')

y_true = hf.get_y_true(shape=(train.samples, len(train.class_indices)), classes=train.classes)

val_stop, val_stop_df = hf.get_flow(df_file=config['data_folder'] + '/val_df.json.bz2',
                          batch_size=config['batch_size'],
                          image_dimension=config['image_dimension'])
print(f'val_stop_df shape: {val_stop_df.shape}')
print('LOG: Got the validation flow')

# Load human-labeled set
def get_human_flow(human_df_address):
    human_df = pd.read_parquet(human_df_address)
    human_df['labels'] = human_df.apply(lambda x: list(x.labels), axis=1) # otherwise the labels column will be a list of lists
    human_df['url'] = human_df.apply(lambda x: '/scratch/WIT_Dataset/images/' + x.url.split('/wikipedia/commons/')[1], axis=1)
    human_df['url'] = human_df['url'].apply(lambda encoded_filename : urllib.parse.unquote(encoded_filename).encode().decode('unicode-escape'))
    print(f'----------------------- \nhuman_df shape: {human_df.shape}\n\n\n')
    human, _ = hf.get_flow(df=human_df, batch_size=config['batch_size'], image_dimension=config['image_dimension'])
    return human
human = get_human_flow('../../data/evaluation/annotated_validation.parquet')

# ============================================
# ============= CREATE MODEL =================
# ============================================
print('LOG: creating and training model')
start = time.time()
model = hf.create_model(n_labels=len(train.class_indices), 
                        image_dimension=config['image_dimension'],
                        random_initialization=config['random_initialization'],
                        loss=config['loss_function'],
                        y_true=y_true)


# ============================================
# ================ TRAIN MODEL ===============
# ============================================

# Save model in-between epochs
checkpoint_path = config['results_folder'] + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=False,
                                                 monitor='val_pr_auc', 
                                                 mode='max',
                                                 save_weights_only=True,
                                                 period=5,
                                                 verbose=1)

history_callback = tf.keras.callbacks.CSVLogger(f"{config['results_folder']}/history.csv", separator=',', append=True)
evaluate_callback = EvaluateCallback(human_dataflow=human, val_dataflow=val_stop, history_csv_path=f"{config['results_folder']}/history_human.csv")

if config['class_weights'] == True:
    class_weights = hf.compute_class_weights(y_true)
    history = model.fit(
        train,
        verbose=2, # one line per epoch 
        validation_data=val_stop,
        epochs=config['epochs'],
        callbacks=[history_callback, evaluate_callback],
        class_weight=class_weights)
else:
    print('\n\nStarting training...')
    history = model.fit(
        train,
        verbose=2, # one line per epoch 
        epochs=config['epochs'],
        callbacks=[cp_callback, history_callback, evaluate_callback])
hf.print_time(start)

print('\n\nTotal training time:')
hf.print_time(start_all)