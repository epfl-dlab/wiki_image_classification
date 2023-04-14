import datetime

configs = []


# ------------- 230412_flatModel_sampleWeight, 15 epochs
config = {}
# Training hyper parameters
config['batch_size'] = 128
config['epochs'] = 15
config['image_dimension'] = 128 #(height and width to which all images will be resized)
config['monitor'] = 'loss'
# Techniques
config['random_initialization'] = False
config['class_weights'] = False
config['hierarchical'] = False
config['loss_function'] = 'sample_weight' # focal_loss, binary_crossentropy, or sample_weight
config['number_trainable_layers'] = 'all' #(number of trainable layers of the basemodel. either 'all' or an integer)
# Folders
config['data_folder'] = 'data/split_data_230412' #(path to folder where the train_df.json.bz2 and val_df.json.bz2 are in)
now = datetime.datetime.now()
YEAR = str(now.year)[-2:]
MONTH = str(now.month).zfill(2)
DAY = str(now.day).zfill(2)
config['results_folder'] = f"results_paper/230412_flatModel_sampleWeights" #(path to folder where training numbers will be saved)

configs.append(config)



# ------------- 230414_flatModel_sample_weight_monitorPRAUC_30epochs,  30 epochs, monitor pr_auc
config = {}
# Training hyper parameters
config['batch_size'] = 128
config['epochs'] = 30
config['image_dimension'] = 128 #(height and width to which all images will be resized)
config['monitor'] = 'pr_auc'

# Techniques
config['random_initialization'] = False
config['class_weights'] = False
config['hierarchical'] = False
config['loss_function'] = 'sample_weight' # focal_loss, binary_crossentropy, or sample_weight
config['number_trainable_layers'] = 'all' #(number of trainable layers of the basemodel. either 'all' or an integer)
# Folders
config['data_folder'] = 'data/split_data_230412' #(path to folder where the train_df.json.bz2 and val_df.json.bz2 are in)
now = datetime.datetime.now()
YEAR = str(now.year)[-2:]
MONTH = str(now.month).zfill(2)
DAY = str(now.day).zfill(2)
config['results_folder'] = f"results_paper/230414_flatModel_{config['loss_function']}_monitor{config['monitor']}" #(path to folder where training numbers will be saved)

configs.append(config)
