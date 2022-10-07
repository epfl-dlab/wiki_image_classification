import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from focal_loss import BinaryFocalLoss
from sklearn.metrics import f1_score
import seaborn as sns
from matplotlib.colors import LogNorm

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

    # # Prediction probabilities for ground-truth TRUE and FALSE
    # fig, axs = plt.subplots(5, 4, figsize=(12, 12))
    # fig.tight_layout(h_pad=3.0, w_pad=3.0)
    # bins = np.linspace(0, 1, 75)
    # for label_idx, ax in zip(range(len(labels)), axs.flatten()):
    #     ax.hist(probs[y_true[:, label_idx] == 0][:, label_idx], bins, alpha=0.5, label='false', log=True)
    #     ax.hist(probs[y_true[:, label_idx] == 1][:, label_idx], bins, alpha=0.5, label='true', log=True)
    #     ax.axvline(x=best_thresholds[label_idx], color='k', linestyle='--')
    #     ax.legend(loc='upper right')
    #     ax.set_title(labels[label_idx])
    #     ax.set_xlabel('Threshold')
    #     ax.set_ylabel('Count')


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


def imbalance_ratio(y_true, class_names):
    """
    Computes the unweighted imbalance ratio (IR) of the dataset.
    IR = N_majority / N_minority
    In our specific case, the majority class is 0s and the minority class is 1s.
    
    Input:  y_true - ground-truth array of format (nr_images x nr_classes)
    Output: imbalance_ratio - integer
    """
    sum_of_1s = y_true.sum(axis=0)
    sum_of_0s = y_true.shape[0] - y_true.sum(axis=0)
    per_class_ir = sum_of_0s / sum_of_1s
    per_class_ir = dict(zip(list(class_names), per_class_ir))
    mean_imbalance_ratio = np.array(list(per_class_ir.values())).mean()
    print(f'Per-class imbalance ratio (IR):\n{per_class_ir}\n')
    print(f'Unweighted mean imbalance ratio: {np.round(mean_imbalance_ratio, 2)}')
    return mean_imbalance_ratio


def undersample(y_true, class_names, df):
    """ 
    Constructs a more balanced test set in a dummy way by adding to it only the images that contain 
    the - for the moment - most uncommon class.
    Inputs:
        - classes: [[]]
        - class_names: list with all labels: 
        - test_df: dataframe with rows containing image files and labels
    """
    sorted_indices = np.argsort(np.sum(y_true, axis=0))
    sorted_class_names = np.array(list(class_names))[sorted_indices]
    least_common_class = sorted_class_names[0]
    balanced_classes = []
    row_ids = []
    counter = 0 
    for index, row in df.iterrows():
        counter += 1
        if counter % 10 == 0:
            y_true = get_y_true(balanced_classes, 40)
            sorted_indices = np.argsort(np.sum(y_true, axis=0))
            sorted_class_names = np.array(class_names)[sorted_indices]
            least_common_class = sorted_class_names[0]
        if least_common_class in row.labels:
            balanced_classes.append(row.labels)
            row_ids.append(index)

    return df.loc[row_ids, :]


def get_flow(df_file, nr_classes, image_dimension):
    df = pd.read_json(df_file, compression='bz2')
    top_classes = get_top_classes(nr_classes, df) 
    # Only keep rows which have either of the top classes
    ids_x_labels = df.labels.apply(lambda classes_list: any([True for a_class in top_classes if a_class in classes_list]))
    df_x_labels = df[ids_x_labels]
    df_x_labels['labels'] = df['labels'].apply(lambda labels_list: [label for label in labels_list if label in top_classes])
    df = df_x_labels.copy()

    datagen = ImageDataGenerator() 
    flow = datagen.flow_from_dataframe(
            dataframe=df, 
            directory='/scratch/WIT_Dataset/images',
            color_mode='rgb',
            x_col='url', 
            y_col='labels', 
            class_mode='categorical', 
            target_size=(image_dimension, image_dimension),
            shuffle=False)
    return flow


def plot_confusion_matrices(confusion_matrix, label_names, data_folder):
    def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
        # From https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes, cmap='YlGnBu', norm=LogNorm())
        except ValueError:
            raise ValueError('Confusion matrix values must be integers.')
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_ylabel('True label', fontsize=8)
        axes.set_xlabel('Predicted label', fontsize=8)
        axes.set_title(class_label)

    fig, ax = plt.subplots(5, 4, figsize=(10, 10))
        
    for axes, cfs_matrix, label in zip(ax.flatten(), confusion_matrix, label_names):
        print_confusion_matrix(cfs_matrix, axes, label, ['N', 'P'])
        
    fig.tight_layout()
    plt.savefig(data_folder + '/confusion_matrix.png')


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