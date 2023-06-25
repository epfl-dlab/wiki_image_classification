import sys
import pandas as pd
import tensorflow as tf
import help_functions as hf
from PIL import PngImagePlugin
import time
hf.setup_gpu(gpu_nr=0)

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

start = time.time()
# ================== HYPER-PARAMETERS ==================
from configs import configs
config = configs[0]
config['epochs'] = 100

old_stdout = sys.stdout
log_file = open(config['results_folder'] + '/log_eval.txt', 'w')
sys.stdout = log_file

# Path to the desired weights:

test, urls = hf.get_flow_urls(
                      df_file=f"{config['data_folder']}/test_df.json.bz2", 
                    #   df_file='data/split_hierarchical_data_221218/10k_samples_from_train.json.bz2', # small file for testing
                      batch_size=config['batch_size'],
                      image_dimension=config['image_dimension'])
assert(test.samples == len(urls))
print('got flow object and urls')
model = hf.create_model(n_labels=len(test.class_indices), image_dimension=config['image_dimension'], number_trainable_layers=config['number_trainable_layers'])

latest = tf.train.latest_checkpoint(config['results_folder'])

print('Predicting on test set:\n')

probs_test = model.predict(test, verbose=2)
print('done predicting model')
print(probs_test.shape)
del model 
print('deleted model object')
mydf = pd.DataFrame(probs_test, columns=list(test.class_indices.keys()))
print('created df')
print(mydf.shape)
del probs_test, test
print('deleted probs test and test')
mydf['url'] = urls
print(mydf.head())

mydf.to_json('results_paper/probs_df_010523.json.bz2', compression='bz2')

hf.print_time(start)