import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from focal_loss import BinaryFocalLoss
from sklearn.metrics import f1_score


def get_optimal_threshold(y_true, probs, thresholds, labels, N=3, strat_size=0.4):
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

    Inspiration: GHOST paper.
    """

    def to_label(probs, threshold):
        return (probs >= threshold) * 1

    nr_images = y_true.shape[0]
    
    best_thresholds = np.zeros((len(labels), N))

    fig, axs = plt.subplots(5, 4, figsize=(12, 12))
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

    fig, axs = plt.subplots(5, 4, figsize=(12, 12))
    fig.tight_layout(h_pad=3.0, w_pad=3.0)
    bins = np.linspace(0, 1, 50)

    for label_idx, ax in zip(range(len(labels)), axs.flatten()):
        ax.hist(probs[y_true[:, label_idx] == 0][:, label_idx], bins, alpha=0.5, label='false', log=True)
        ax.hist(probs[y_true[:, label_idx] == 1][:, label_idx], bins, alpha=0.5, label='true', log=True)
        ax.axvline(x=optim_thresholds[label_idx], color='k', linestyle='--')
        ax.legend(loc='upper right')
        ax.set_title(labels[label_idx])
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Count')

    return optim_thresholds


def plot_probs_and_best_threshold(y_true, probs, labels):
    def to_label(probs, threshold):
        return (probs >= threshold) * 1

    thresholds = np.linspace(start=0, stop=1, num=101)
    
    best_thresholds = np.zeros((len(labels, )))

    # F1-scores per threshold
    fig, axs = plt.subplots(5, 4, figsize=(12, 12))
    fig.tight_layout(h_pad=3.0, w_pad=3.0)
    for label_idx, ax in zip(range(len(labels)), axs.flatten()):
        f1_scores = [f1_score(y_true=y_true[:, label_idx], y_pred=to_label(probs[:, label_idx], t)) for t in thresholds]
        best_thresholds[label_idx] = thresholds[np.argmax(f1_scores)]
        ax.axvline(x=best_thresholds[label_idx], color='k', linestyle='--')
        ax.plot(thresholds, f1_scores)
        ax.set_title(labels[label_idx])
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1-score')

    # Prediction probabilities for ground-truth TRUE and FALSE
    fig, axs = plt.subplots(5, 4, figsize=(12, 12))
    fig.tight_layout(h_pad=3.0, w_pad=3.0)
    bins = np.linspace(0, 1, 75)
    for label_idx, ax in zip(range(len(labels)), axs.flatten()):
        ax.hist(probs[y_true[:, label_idx] == 0][:, label_idx], bins, alpha=0.5, label='false', log=True)
        ax.hist(probs[y_true[:, label_idx] == 1][:, label_idx], bins, alpha=0.5, label='true', log=True)
        ax.axvline(x=best_thresholds[label_idx], color='k', linestyle='--')
        ax.legend(loc='upper right')
        ax.set_title(labels[label_idx])
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Count')

def get_top_classes(nr_classes, df):
    """Returns the nr_classes classes with greater number of samples from the multiclass df."""
    _generator = ImageDataGenerator() 
    _data = _generator.flow_from_dataframe(dataframe=df, 
                                        directory='/scratch/WIT_Dataset/images', 
                                        x_col='url', 
                                        y_col='labels', 
                                        class_mode='categorical', 
                                        validate_filenames=False)

    y_true = get_y_true(shape=(_data.samples, len(_data.class_indices)), classes=_data.classes)

    sorted_indices = np.argsort(np.sum(y_true, axis=0))[::-1]
    return np.array(list(_data.class_indices.keys()))[sorted_indices[:nr_classes]]



def create_model(n_labels, image_dimension, model_name, number_trainable_layers, loss='binary_crossentropy'):
    """Take efficientnet pre-trained on imagenet-1k, not including the last layer."""
    if model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(include_top=False, 
                                    weights='imagenet', 
                                    classes=n_labels,
                                    input_shape=(image_dimension, image_dimension, 3))
    elif model_name == 'EfficientNetB1':
        base_model = EfficientNetB1(include_top=False, 
                                    weights='imagenet', 
                                    classes=n_labels,
                                    input_shape=(image_dimension, image_dimension, 3))
    elif model_name == 'EfficientNetB2':
        base_model = EfficientNetB2(include_top=False, 
                                    weights='imagenet', 
                                    classes=n_labels,
                                    input_shape=(image_dimension, image_dimension, 3))

    print(f'\nNumber of layers in basemodel: {len(base_model.layers)}')
    # Fine tune from this layer onwards
    # fine_tune_at = round((1 - config['percent_trainable_layers']) * len(efficient_net.layers))
    fine_tune_at = len(base_model.layers) - number_trainable_layers
 
    print(f'Number of trainable layers: {len(base_model.layers) - fine_tune_at}\n')
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model = Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_labels, activation='sigmoid')
    ])

    if loss == 'binary_crossentropy':
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'categorical_accuracy'])
    # Binary Focal Cross Entropy
    elif loss == 'binary_focal_crossentropy':
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=BinaryFocalLoss(gamma=2, from_logits=False),
                    metrics=['accuracy', 'categorical_accuracy'])

    model.summary()
    return model


# def get_y_true(samples, class_indices, classes):
#     y_true = np.zeros((samples, len(class_indices))) # nr_rows=nr_images; nr_columns=nr_classes
#     for row_idx, row in enumerate(classes):
#         for idx in row:
#             y_true[row_idx, idx] = 1
#     return y_true

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


def plot_distribution(dataframe, filename, minimal_nr_images=0):
    _generator = ImageDataGenerator() 
    _data = _generator.flow_from_dataframe(dataframe=dataframe, 
                                            directory='/scratch/WIT_Dataset/images', 
                                            x_col='url', 
                                            y_col='labels', 
                                            class_mode='categorical', 
                                            validate_filenames=False)

    y_true = get_y_true(shape=(_data.samples, len(_data.class_indices)), classes=_data.classes)

    sorted_indices = np.argsort(np.sum(y_true, axis=0))
    sorted_images_per_class = y_true.sum(axis=0)[sorted_indices]

    mask_kept = y_true.sum(axis=0)[sorted_indices] > minimal_nr_images
    mask_removed = y_true.sum(axis=0)[sorted_indices] < minimal_nr_images

    _ = plt.figure(figsize=(8, 14))

    _ = plt.barh(np.array(range(y_true.shape[1]))[mask_kept], sorted_images_per_class[mask_kept], color='blue', alpha=0.6)
    _ = plt.barh(np.array(range(y_true.shape[1]))[mask_removed], sorted_images_per_class[mask_removed], color='red', alpha=0.6)

    _ = plt.yticks(range(y_true.shape[1]), np.array(list(_data.class_indices.keys()))[sorted_indices], fontsize=12)
    _ = plt.xscale('log')
    _ = plt.xlabel('Count', fontsize=13)
    _ = plt.ylabel('Labels', fontsize=13)
    _ = plt.grid(True)

    plt.legend(['Kept', 'Removed'], loc='upper right', fontsize=12)
    plt.savefig(filename)