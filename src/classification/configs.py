﻿import datetime

configs = []


# ------------- 100 epochs, sample_weight, pre-trained weights
config = {}
# Training hyper parameters
config['batch_size'] = 2
config['epochs'] = 100
config['image_dimension'] = 32 #(height and width to which all images will be resized)
config['monitor'] = 'pr_auc'

# Techniques
config['random_initialization'] = False 
config['class_weights'] = False
config['hierarchical'] = False
config['loss_function'] = 'sample_weight' # binary_crossentropy, or sample_weight
# Folders
config['data_folder'] = 'data/jpg-data' #(path to folder where the train_df.json.bz2 and val_df.json.bz2 are in)
config['results_folder'] = f"results_paper/230621_pretrained{not config['random_initialization']}_{config['loss_function']}_{config['epochs']}epochs" #(path to folder where training numbers will be saved)

configs.append(config)


# ------------- 100 epochs, binary_crossentropy, pre-trained weights
config = {}
# Training hyper parameters
config['batch_size'] = 32
config['epochs'] = 100
config['image_dimension'] = 128 #(height and width to which all images will be resized)
config['monitor'] = 'pr_auc'

# Techniques
config['random_initialization'] = False 
config['class_weights'] = False
config['hierarchical'] = False
config['loss_function'] = 'binary_crossentropy' # binary_crossentropy, or sample_weight
# Folders
config['data_folder'] = 'data/split_data_150623' #(path to folder where the train_df.json.bz2 and val_df.json.bz2 are in)
config['results_folder'] = f"results_paper/230619_pretrained{not config['random_initialization']}_{config['loss_function']}_{config['epochs']}epochs" #(path to folder where training numbers will be saved)

configs.append(config)

