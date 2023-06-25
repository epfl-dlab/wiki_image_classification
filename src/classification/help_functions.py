#
# This file contains a variety of help functions used in the train_classification.py file, and in the 
# Jupyter notebooks. 
#

# Standard Python libraries
import os
import time
import numpy as np
import pandas as pd
from functools import partial
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Tensorflow & Keras
import tensorflow as tf
from keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB2, EfficientNetB1 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Other stuff
from focal_loss import BinaryFocalLoss


# =================================== Model-related =========================================

def create_model(n_labels, image_dimension, y_true=None, loss='binary_crossentropy', random_initialization=False):
    """'
    Take EfficientNetB2 pre-trained on imagenet-1k, not including the last layer.
    """
    if random_initialization:
        base_model = EfficientNetB2(include_top=False, 
                                    weights=None, 
                                    classes=n_labels,
                                    input_shape=(image_dimension, image_dimension, 3))
    else:
        base_model = EfficientNetB2(include_top=False, 
                                    weights='imagenet', 
                                    classes=n_labels,
                                    input_shape=(image_dimension, image_dimension, 3))

    print(f'\nNumber of layers in basemodel: {len(base_model.layers)}')

    base_model.trainable = True # TODO: works?
 
    model = Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_labels, activation='sigmoid')
    ])
    # Standard binary cross entropy loss
    if loss == 'binary_crossentropy':
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='binary_crossentropy',
                    metrics=[
                        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(num_thresholds=50, curve='PR', name='pr_auc', multi_label=True),
                    ])
    # Binary focal cross entropy loss
    elif loss == 'focal_loss':
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=BinaryFocalLoss(gamma=2, from_logits=False),
                    metrics=[
                        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(num_thresholds=50, curve='PR', name='pr_auc', multi_label=True),
                    ])
    # Sample weight loss
    elif loss == 'sample_weight':
        positive_samples_per_class = np.sum(y_true, axis=0)
        negative_samples_per_class = y_true.shape[0] - positive_samples_per_class
        alpha_weights = negative_samples_per_class / positive_samples_per_class
        alpha_weights = tf.cast(alpha_weights, tf.float32)
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss=get_custom_loss(alpha_weights),
                        metrics=[
                            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall'),
                            tf.keras.metrics.AUC(num_thresholds=50, curve='PR', name='pr_auc', multi_label=True),
                            tf.keras.metrics.AUC(num_thresholds=50, curve='ROC', name='roc_auc', multi_label=True)
                        ])
    else:
        raise ValueError('This loss function is not supported!')

    model.summary()
    return model


def get_flow(image_dimension, batch_size, df_file='', df=None):
    if df_file:
        df = pd.read_json(df_file, compression='bz2')
    datagen = ImageDataGenerator() 
    flow = datagen.flow_from_dataframe(
            dataframe=df, 
            # directory='/scratch/WIT_Dataset/images',
            color_mode='rgb',
            batch_size=batch_size,
            x_col='url', 
            y_col='labels', 
            class_mode='categorical', 
            target_size=(image_dimension, image_dimension),
            shuffle=False)
    return flow, df


# fast fix for evaluate.py of 4M images. return df.url.values instead of the whole df
def get_flow_urls(image_dimension, batch_size, df_file='', df=None):
    if df_file:
        df = pd.read_json(df_file, compression='bz2')
    datagen = ImageDataGenerator() 

    # --------------- not sure why I added this, probably some file was failing at prediction
    white_list_formats = ("png", "jpg", "jpeg", "bmp", "ppm", "tif", "tiff")
    def validate_filename(filename, white_list_formats):
        return filename.lower().endswith(white_list_formats) and os.path.isfile(filename)

    filepaths = df['url'].map(lambda fname: os.path.join('/scratch/WIT_Dataset/images', fname))
    mask = filepaths.apply(validate_filename, args=(white_list_formats,))
    df = df[mask]
    # ---------------------------------------------------------------------------------------

    flow = datagen.flow_from_dataframe(
            dataframe=df, 
            # directory='/scratch/WIT_Dataset/images',
            color_mode='rgb',
            batch_size=batch_size,
            x_col='url', 
            y_col='labels', 
            class_mode='categorical', 
            target_size=(image_dimension, image_dimension),
            validate_filenames=False,
            shuffle=False)
    return flow, df.url.values 


def get_optimal_threshold(y_true, probs, thresholds, labels, image_path, N=3):
    """
    Calculates the best threshold per class by calculating the median of the optimal thresholds
    over stratified subsets of the training set.

    Input:  N          - number of stratifications
            strat_size - size of the stratification, between 0.05 and 0.5
            y_true     - binary matrix with the ground-truth values (nr_images, nr_labels)
            probs      - matrix containing the prediction probabilities (nr_images, nr_labels)
            thresholds - possible thresholds (e.g. [0.05, 0.10, 0.15, ..., 0.95])
            labels     - labes [Art, Architecture, ...]
    Output: best_thresholds (nr_labels, )

    Inspiration: "GHOST: Adjusting the Decision Threshold to Handle Imbalanced Data in Machine Learning" by Esposito et al.
    """

    def to_label(probs, threshold):
        return (probs >= threshold) * 1

    nr_images = y_true.shape[0]
    
    best_thresholds = np.zeros((len(labels), N))

    fig, axs = plt.subplots(8, 4, figsize=(12, 12))
    fig.tight_layout(h_pad=3.0, w_pad=3.0)

    for i in range(N):
        subset_indices = np.random.choice(a=np.arange(nr_images), size=int(np.round(nr_images*0.2)), replace=False)

        for label_idx, ax in zip(range(len(labels)), axs.flatten()):
            f1_scores = [f1_score(y_true=y_true[subset_indices, label_idx], y_pred=to_label(probs[subset_indices, label_idx], t)) for t in thresholds]
            best_thresholds[label_idx, i] = thresholds[np.argmax(f1_scores)]
            # ax.axvline(x=best_thresholds[label_idx, i], color='k', linestyle='--')
            ax.plot(thresholds, f1_scores)
            ax.set_title(labels[label_idx])
            ax.set_xlabel('Threshold')
            ax.set_ylabel('F1-score')

    optim_thresholds = np.median(best_thresholds, axis=1)

    for label_idx, ax in zip(range(len(optim_thresholds)), axs.flatten()):
        ax.axvline(x=optim_thresholds[label_idx], color='k', linestyle='--')

    save_img(image_path + '/optimal_threshold.png')

    fig, axs = plt.subplots(8, 4, figsize=(12, 12))
    fig.tight_layout(h_pad=3.0, w_pad=3.0)
    bins = np.linspace(0, 1, 50)

    for label_idx, ax in zip(range(len(labels)), axs.flatten()):
        ax.hist(probs[y_true[:, label_idx] == 0][:, label_idx], bins, alpha=0.5, label='false', log=True)
        ax.hist(probs[y_true[:, label_idx] == 1][:, label_idx], bins, alpha=0.5, label='true', log=True)
        ax.axvline(x=optim_thresholds[label_idx], color='k', linestyle='--')
        ax.legend(loc='upper right')
        ax.set_title(labels[label_idx])
        ax.set_xlabel('Probability')
        ax.set_ylabel('Count')

    save_img(image_path + '/probs.png')

    return optim_thresholds
      

def get_y_true(shape, classes):
    """
    Gets the ground-truth in a binary matrix format from the sparse matrix given by ImageGenerator.
    Input:  shape   - (nr_images x nr_labels)
            classes - sparse matrix (array of arrays) given by ImageGenerator
    Output: y_true  - binary matrix of dimension shape
    """
    y_true = np.zeros(shape) # nr_rows=nr_images; nr_columns=nr_classes
    for row_idx, row in enumerate(classes):
        for idx in row:
            y_true[row_idx, idx] = 1
    return y_true


def compute_class_weights(y_true):
    """
    Computes class_weights to compensate imbalanced classes. Inspired in 
    'https://towardsdatascience.com/dealing-with-imbalanced-data-in-tensorflow-class-weights-60f876911f99'.
    Dictionary mapping class indices (integers) to a weight (float) value, 
    used for weighting the loss function (during training only).
    """
    class_count = y_true.sum(axis=0)
    n_samples = y_true.shape[0] 
    n_classes = y_true.shape[1]

    # Compute class weights using balanced method
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights))
    return dict(zip(class_labels, class_weights))


def get_custom_loss(alpha_weights):
    """
    Sample weight loss implementation, inspired by https://www.tensorflow.org/guide/keras/train_and_evaluate#sample_weights.

    Input:  
        - alpha_weights: a Tensorflow tensor with dimension (n_labels x 1) containing the n_labels positive weights of each label.
                         Cast it to a Tensorflow tensor with: `alpha_weights = tf.cast(alpha_weights, tf.float32)`.
    Output:
        - custom_loss: a function, to be used as https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy.
    """
    epsilon = 1e-7
    @tf.function
    def custom_loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.cast(y_pred, y_pred.dtype)

        bce = y_true * tf.math.log(y_pred + epsilon) * alpha_weights
        bce += (1 - y_true) * tf.math.log(1 - y_pred + epsilon)

        return -backend.mean(bce)
    return custom_loss

# ====================================== Metrics ============================================

def imbalance_ratio_per_label(y_true):
    """
    Returns the IRLbl for all labels. Metric introduced in Addressing imbalance in multilabel classification
    by Charte et al.. The greater the IRLbl, the greater the imbalance in the MLD.

    The formula is: IR(label_i) = nr_samples(label_with_most_samples) / nr_samples(label_i)

    Input: y_true - binary ground-truth array of format (nr_images x nr_classes)
    Output: IRLbl for all classes.
    """
    return np.max(y_true.sum(axis=0)) / y_true.sum(axis=0)


def mean_imbalance_ratio(y_true, class_names=[]):
    """
    Computes the mean imbalance ratio (meanIR) of the dataset as defined in Charte et al. 
    2015.
    
    Input:  y_true - ground-truth array of format (nr_images x nr_classes)
    Output: imbalance_ratio
    """
    IRLbl = imbalance_ratio_per_label(y_true)
    if class_names:
        ir_dict = dict(zip(list(class_names), np.round(IRLbl, 2)))
        # print(ir_dict)
    return np.sum(IRLbl) / len(IRLbl), ir_dict


def scumble(y_true):
    """
    SCUMBLE (Score of ConcUrrence among iMBalanced LabEls), metric introduced by Charte
    et al. in "Dealing with Difficult Minority Labels in Imbalanced Mutilabel Data Sets".

    The output is between 0 and 1; the smaller it is, the less concurrent the labels are,
    the greater it is, the more concurrent are the labels.
    
    Output: 
        SCUMBLE_D   - integer average SCUMBLE of all instances.
        SCUMBLE_ins - SCUMBLE metric for every instance
    """
    IRLbl = imbalance_ratio_per_label(y_true)
    L = y_true.shape[1] # nr_classes
    prod = IRLbl * y_true
    IRLbl_bar = prod.sum(axis=1) / np.count_nonzero(prod, axis=1) # IRLbl bar, in eq. (3)
    SCUMBLE_ins = 1 - (1/IRLbl_bar) * np.power(np.prod(prod, where=[prod != 0], axis=1), 1/L)
    SCUMBLE_D = SCUMBLE_ins.mean() # eq. (4)
    return SCUMBLE_D, SCUMBLE_ins


def get_metrics(y_true, y_pred, label_names, image_path):
    """
    Prints F1-score, precision, and recall for all classes, top 5 majority classes, and the rest minority classes.

    Output:
        - f1-scores for all classes.
    """
    print(f'\nMean number of label assignments per image in ground-truth: {np.sum(y_true) / y_true.shape[0]:.4f}')
    print(f'Mean number of label assignments per image in predictions: {np.sum(y_pred) / y_pred.shape[0]:.4f}\n')

    n_labels = y_pred.shape[1]
    metrics_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=label_names, output_dict=True)).transpose()
    metrics_df['index'] = np.concatenate((np.arange(start=0, stop=n_labels), [None, None, None, None]))
    print(metrics_df)

    # Sort the classes by size, descending order
    sorted_class_sizes = np.argsort(np.sum(y_true, axis=0))[::-1]
    print(f'Ordered class in descending order of samples:\n {np.array(label_names)[sorted_class_sizes]}')

    # F1-scores
    sorted_f1score_per_class = metrics_df['f1-score'][0:n_labels][sorted_class_sizes]

    print(f'\nUnweighted avg. F1-score of all classes: {np.sum(sorted_f1score_per_class) / n_labels}')
    print(f'Unweighted avg. F1-score of top 5 classes: {np.sum(sorted_f1score_per_class[:5]) / 5}')
    print(f'Unweighted avg. F1-score of the rest: {np.sum(sorted_f1score_per_class[5:]) / (n_labels - 5)}\n')

    # Precision
    sorted_precision_per_class = metrics_df['precision'][0:n_labels][sorted_class_sizes]

    print(f'\nUnweighted avg. precision of all classes: {np.sum(sorted_precision_per_class) / n_labels}')
    print(f'Unweighted avg. precision of top 5 classes: {np.sum(sorted_precision_per_class[:5]) / 5}')
    print(f'Unweighted avg. precision of the rest: {np.sum(sorted_precision_per_class[5:]) / (n_labels - 5)}\n')

    # Recall
    sorted_recall_per_class = metrics_df['recall'][0:n_labels][sorted_class_sizes]

    print(f'\nUnweighted avg. recall of all classes: {np.sum(sorted_recall_per_class) / n_labels}')
    print(f'Unweighted avg. recall of top 5 classes: {np.sum(sorted_recall_per_class[:5]) / 5}')
    print(f'Unweighted avg. recall of the rest: {np.sum(sorted_recall_per_class[5:]) / (n_labels - 5)}\n')

    if image_path:
        _ = plt.figure(figsize=(8, 14))
                        
        _ = plt.title('F1-score per class')
        _ = plt.barh(range(y_true.shape[1]), sorted_f1score_per_class, color='blue', alpha=0.6)
        _ = plt.yticks(ticks=range(n_labels), labels=np.array(label_names)[sorted_class_sizes])
        _ = plt.xlabel('F1-score')
        _ = plt.grid(True)
        save_img(image_path)

    return metrics_df['f1-score'][0:n_labels]


# ====================================== Utilities ==========================================

def setup_gpu(gpu_nr):
    """
    Limit the code to run on the GPU with number gpu_nr (0 or 1 in iccluster039). 
    """
    # tf.config.threading.set_intra_op_parallelism_threads(10) 
    # tf.config.threading.set_inter_op_parallelism_threads(10) 
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[gpu_nr], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)


def print_time(start, ms=False):
    end = time.time()
    try:
        if ms:
            total_time_in_ms = round((end - start) * 1000, 3)
            print(f'Time passed: {total_time_in_ms} ms\n')
        else:
            total_time_in_hours = round((end - start) / 3600, 2)
            print(f'Time passed: {total_time_in_hours} hours')
            print(time.strftime("%H:%M:%S", time.localtime()))
    except:
        print('failed to print time')


def save_img(image_path):
    try:
        plt.savefig(image_path, bbox_inches='tight')
    except:
        print(f'ERROR: Could not save image {image_path}')


def plot_distribution(dataframe, filename):
    _generator = ImageDataGenerator() 
    _data = _generator.flow_from_dataframe(dataframe=dataframe, 
                                            # directory='/scratch/WIT_Dataset/images',  # TODO: remove this. url will have an absolute path
                                            x_col='url', 
                                            y_col='labels', 
                                            class_mode='categorical', 
                                            validate_filenames=False)

    y_true = get_y_true(shape=(_data.samples, len(_data.class_indices)), classes=_data.classes)

    sorted_indices = np.argsort(np.sum(y_true, axis=0))
    sorted_images_per_class = y_true.sum(axis=0)[sorted_indices]

    _ = plt.figure(figsize=(8, 14))
    _ = plt.barh(np.array(range(y_true.shape[1])), sorted_images_per_class, color='blue', alpha=0.6)
    _ = plt.yticks(range(y_true.shape[1]), np.array(list(_data.class_indices.keys()))[sorted_indices], fontsize=12)
    _ = plt.xscale('log')
    _ = plt.xlabel('Count', fontsize=13)
    _ = plt.ylabel('Labels', fontsize=13)
    _ = plt.grid(True)
    save_img(filename)


def print_file_percentages_in_url(some_df):
    """The some_df dataframe has a url column whose file ending we compute percentages for."""
    # Count the number of occurrences of each file-ending in the url column
    counts = some_df['url'].str.lower().str.extract(r'\.([a-z]+)$', expand=False).value_counts()

    # Divide each count by the total number of rows and multiply by 100 to get the percentage
    percentages = counts / len(some_df) * 100

    # Print the percentages for each file-ending
    print(percentages)