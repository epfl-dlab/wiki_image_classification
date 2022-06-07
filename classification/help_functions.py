from distutils.log import error
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_top_classes(nr_classes, df):
    """Returns the nr_classes classes with greater number of samples from the multiclass df."""
    _generator = ImageDataGenerator() 
    _data = _generator.flow_from_dataframe(dataframe=df, 
                                        directory='/scratch/WIT_Dataset/images', 
                                        x_col='url', 
                                        y_col='labels', 
                                        class_mode='categorical', 
                                        validate_filenames=False)

    y_true = get_y_true(_data.samples, _data.class_indices, _data.classes)

    sorted_indices = np.argsort(np.sum(y_true, axis=0))[::-1]
    return np.array(list(_data.class_indices.keys()))[sorted_indices[:nr_classes]]


def create_model(n_labels, image_dimension, model_name='EfficientNetB0'):
    """Take efficientnet pre-trained on imagenet-1k, not including the last layer."""
    if model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(include_top=False, 
                                    weights='imagenet', 
                                    classes=n_labels,
                                    input_shape=(image_dimension, image_dimension, 3))
    else:
        raise error

    base_model.trainable=False

    model = Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_labels, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def get_y_true(samples, class_indices, classes):
    y_true = np.zeros((samples, len(class_indices))) # nr_rows=nr_images; nr_columns=nr_classes
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

# def get_y_true(classes, preset_nr_classes=0):
#     """Gets one-hot encoded matrix of format (nr_images)x(nr_classes)."""
#     nr_images = len(classes)
#     if not preset_nr_classes:
#         nr_classes = len(set([item for sublist in classes for item in sublist]))
#     else:
#         nr_classes = preset_nr_classes
#     y_true = np.zeros((nr_images, nr_classes))
#     for row_idx, row in enumerate(classes):
#         for idx in row:
#             y_true[row_idx, idx] = 1
#     return y_true

def balance_test(classes, class_names, test_df):
        """ 
        Constructs a sort of more balanced test set in a dummy way by adding to it only the images that contain 
        the - for the moment - most uncommon class.
        Inputs:
            - classes: [[]]
            - class_names: list with all labels: 
            - test_df: dataframe with rows containing image files and labels
        """
        y_true = get_y_true(classes)
        sorted_indices = np.argsort(np.sum(y_true, axis=0))
        sorted_class_names = np.array(list(class_names))[sorted_indices]
        least_common_class = sorted_class_names[0]
        balanced_classes = []
        row_ids = []
        counter = 0 
        for index, row in test_df.iterrows():
            counter += 1
            if counter % 10 == 0:
                y_true = get_y_true(balanced_classes, 40)
                sorted_indices = np.argsort(np.sum(y_true, axis=0))
                sorted_class_names = np.array(class_names)[sorted_indices]
                least_common_class = sorted_class_names[0]
            if least_common_class in row.labels:
                balanced_classes.append(row.labels)
                row_ids.append(index)

        return test_df.loc[row_ids, :]
    
def plot_distribution(classes, class_names=[], description='None', plot_log=False, lims=()):
    y_true = get_y_true(classes)
    sorted_indices = np.argsort(np.sum(y_true, axis=0))
    sorted_images_per_class = y_true.sum(axis=0)[sorted_indices]
    plt.figure(figsize=(12, 15))
    plt.title('Number of images per class' + description)
    if plot_log:
        plt.xscale('log')
    if lims:
        plt.xlim(lims)
    plt.xlabel('Count')
    plt.grid(True)
    plt.barh(np.array(range(y_true.shape[1])), sorted_images_per_class, color='blue', alpha=0.65)
    if class_names:
        plt.yticks(range(y_true.shape[1]), np.array(list(class_names))[sorted_indices])
