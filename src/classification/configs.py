import datetime

configs = []


# ------------- 100 epochs, sample_weight, pre-trained weights
config = {}
# Training hyper parameters
config['batch_size'] = 64
config['epochs'] = 100
config['image_dimension'] = 64 #(height and width to which all images will be resized)
config['monitor'] = 'pr_auc'

# Techniques
config['random_initialization'] = False 
config['class_weights'] = False
config['hierarchical'] = False
config['loss_function'] = 'sample_weight' # binary_crossentropy, or sample_weight
# Folders
config['data_folder'] = '/home/matvieir/wiki_image_classification/src/classification/data/jpg-data' #(path to folder where the train_df.json.bz2 and val_df.json.bz2 are in)
config['results_folder'] = f"results_paper/230703_{config['loss_function']}_{config['epochs']}epochs" #(path to folder where training numbers will be saved)

configs.append(config)


# ------------- 100 epochs, binary_crossentropy, pre-trained weights
config = {}
# Training hyper parameters
config['batch_size'] = 64
config['epochs'] = 100
config['image_dimension'] = 64 #(height and width to which all images will be resized)
config['monitor'] = 'pr_auc'

# Techniques
config['random_initialization'] = False 
config['class_weights'] = False
config['hierarchical'] = False
config['loss_function'] = 'binary_crossentropy' # binary_crossentropy, or sample_weight
# Folders
config['data_folder'] = '/home/matvieir/wiki_image_classification/src/classification/data/jpg-data' #(path to folder where the train_df.json.bz2 and val_df.json.bz2 are in)
config['results_folder'] = f"results_paper/230703_{config['loss_function']}_{config['epochs']}epochs" #(path to folder where training numbers will be saved)

configs.append(config)

# ------------- 100 epochs, binary_crossentropy, pre-trained weights, class weights
config = {}
# Training hyper parameters
config['batch_size'] = 64
config['epochs'] = 100
config['image_dimension'] = 64 #(height and width to which all images will be resized)
config['monitor'] = 'pr_auc'

# Techniques
config['random_initialization'] = False 
config['class_weights'] = True
config['hierarchical'] = False
config['loss_function'] = 'binary_crossentropy' # binary_crossentropy, or sample_weight
# Folders
config['data_folder'] = '/home/matvieir/wiki_image_classification/src/classification/data/jpg-data' #(path to folder where the train_df.json.bz2 and val_df.json.bz2 are in)
config['results_folder'] = f"results_paper/230703_{config['loss_function']}_classweights_{config['epochs']}epochs" #(path to folder where training numbers will be saved)

configs.append(config)