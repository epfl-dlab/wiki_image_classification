# Common libraries
import pandas as pd
import numpy as np
import urllib.parse
import time
import os
from matplotlib import pyplot as plt
from datetime import datetime

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

# Tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten


# Settings
IMAGE_DIMENSION = 64


class DataLoader:
    def __init__(self, filename, min_nr_images_per_class, image_dimension, plot_distribution=False):
        self.filename = filename
        self.image_dimension = image_dimension
        self.min_nr_images_per_class = min_nr_images_per_class
        self.plot_distribution = plot_distribution
        print('Loading dataframe...')
        self.data = self.load_dataframe()
        print('Finished loading dataframe! Started cleaning dataframe...')
        self.data = self.clean_dataframe(self.data)
        print(f'Finished loading and cleaning dataframe! It has {self.data.shape[0]} images.')

    def load_dataframe(self):
        """
        Load dataframe containing labels for each image, update its encoding and
        parses the image path in the folder /scratch/WIT_Dataset/images folder.
        """
        df = pd.read_json(self.filename, compression='bz2')
        # Make sure all images have '/commons/' in their path, otherwise they apparently aren't in the WIT_Dataset
        df = df[df.image_url.str.contains('/commons/')]
        # After /commons/ comes the file location as it is organized in the WIT_dataset
        df['image_path'] = [url.split('commons/')[1] for url in df.image_url]
        # Decode filename paths so they are validated by tensorflow later
        df['image_path'] = df['image_path'].apply(lambda encoded_filename : urllib.parse.unquote(encoded_filename))
        return df

    def get_y_true(self, samples, class_indices, classes):
        """Gets one-hot encoded matrix of format (nr_images)x(nr_classes)."""
        y_true = np.zeros((samples, len(class_indices))) # nr_rows=nr_images; nr_columns=nr_classes
        for row_idx, row in enumerate(classes):
            for idx in row:
                y_true[row_idx, idx] = 1
        return y_true
    
    def clean_dataframe(self, df):
        """
        Removes categories with less than min_nr_images_per_class from the dataframe self.data, 
        and removes the images that were only belonging to these classes from the data dataframe.
        Plots the distribution of images per class (the red ones are the removed).
        """
        _generator = ImageDataGenerator(rescale=1/255, fill_mode='nearest') 
        _data = _generator.flow_from_dataframe(dataframe=self.data, 
                                               directory='/scratch/WIT_Dataset/images', 
                                               x_col='image_path', 
                                               y_col='taxo_labels', 
                                               class_mode='categorical', 
                                               validate_filenames=False, 
                                               target_size=(self.image_dimension, self.image_dimension))
        y_true = self.get_y_true(_data.samples, _data.class_indices, _data.classes)
        indices_of_classes_to_remove = np.where(np.sum(y_true, axis=0) < self.min_nr_images_per_class)
        classes_to_remove = np.array(list(_data.class_indices.keys()))[indices_of_classes_to_remove] 
        df['taxo_labels'] = df['taxo_labels'].apply(lambda labels: [el for el in labels if el not in classes_to_remove])
        
        if self.plot_distribution:
            sorted_indices = np.argsort(np.sum(y_true, axis=0))
            sorted_images_per_class = y_true.sum(axis=0)[sorted_indices]
            print('Number of images per class')
            print(sorted_images_per_class)
            mask_kept = y_true.sum(axis=0)[sorted_indices] > self.min_nr_images_per_class
            mask_removed = y_true.sum(axis=0)[sorted_indices] < self.min_nr_images_per_class
            plt.figure(figsize=(12, 15))
            plt.title('Number of images per class (log-scale on x-axis)')
            plt.barh(np.array(range(y_true.shape[1]))[mask_kept], sorted_images_per_class[mask_kept], color='blue', alpha=0.65)
            plt.barh(np.array(range(y_true.shape[1]))[mask_removed], sorted_images_per_class[mask_removed], color='red', alpha=0.65)
            plt.yticks(range(y_true.shape[1]), np.array(list(_data.class_indices.keys()))[sorted_indices])
            plt.xscale('log')
            plt.xlabel('Count')
            plt.grid(True)
            plt.legend(['Kept', 'Removed'], loc='upper right')
            plt.savefig('results/class_distribution.png')

        return df


class DataSeparator:
    def __init__(self, image_label_df, oversampling=False):

        self.train_df, self.test_df = train_test_split(image_label_df, test_size=0.1, random_state=17)
        self.train_df, self.val_df = train_test_split(self.train_df, train_size=0.9, random_state=17)

        # Data generator for training and validation sets
        if oversampling:
            train_generator = ImageDataGenerator(validation_split=0.10,  rescale=1/255,
                                                 rotation_range=40,      width_shift_range=0.2,
                                                 height_shift_range=0.2, shear_range=0.2,
                                                 zoom_range=0.2,         horizontal_flip=True, fill_mode='nearest') 
        else:
            train_generator = ImageDataGenerator(validation_split=0.10, rescale=1/255)

        print('\n----------- Train images -----------')
        self.train = train_generator.flow_from_dataframe(dataframe=self.train_df,  directory='/scratch/WIT_Dataset/images', 
                                                         x_col='image_path',       y_col='taxo_labels', 
                                                         class_mode='categorical', subset='training',
                                                         validate_filenames=True,  target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION)) # the dimensions to which all images found will be resized.

        print('\n----------- Validation images -----------')          
        self.val = train_generator.flow_from_dataframe(dataframe=self.val_df,    directory='/scratch/WIT_Dataset/images', 
                                                       x_col='image_path',       y_col='taxo_labels', 
                                                       class_mode='categorical', subset='validation',
                                                       validate_filenames=True,  target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))

        # Data generator for test set
        print('\n----------- Test images -----------')
        # balanced_test_df = self.balance_test(test_df)    
        self.test = self.test_df
        
    def balance_test(self, test_df):
        # TODO!

        pass


class ModelTrainer:
    def __init__(self, train, val, epochs=50, use_class_weights=False):
        self.train = train
        self.val = val
        self.n_classes = len(train.class_indices)
        self.model = self.construct_model()
        self.epochs = epochs
        history = self.train_model()
        self.plot_history(history)

    def construct_model(self):
        """
        Construct model in a transfer learning manner: use all weights of the EfficientNetB0 network pre-trained on 
        ImageNet-1k, and add a fully connect layer plus an output layer (with the sigmoid function as activation function). 
        - EfficientNet: https://keras.io/api/applications/efficientnet/
        - Sigmoid activation function on a multi-label classification problem: https://towardsdatascience.com/multi-label-image-classification-with-neural-network-keras-ddc1ab1afede
        """
        efficient_net = EfficientNetB0(include_top=False, weights='imagenet', classes=self.n_classes,
                                       input_shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3))
        efficient_net.trainable = False
        model = Sequential()
        model.add(efficient_net)
        model.add(Flatten())
        model.add(Dense(units=120, activation='relu'))
        model.add(Dense(units=self.n_classes, activation='sigmoid')) # output layer
        model.summary()
        return model

    def train_model(self):
        """
        Compile and train model using the binary cross entropy as loss function.
        - On what loss function to choose: https://stats.stackexchange.com/questions/260505/should-i-use-a-categorical-cross-entropy-or-binary-cross-entropy-loss-for-binary
        """
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy']) 
        history = self.model.fit(self.train, epochs=self.epochs, steps_per_epoch=15, 
                                 validation_data=self.val, validation_steps=7,
                                 verbose=2)
        return history
    
    def plot_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(self.epochs)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(-0.01, 0.4)
        plt.grid(True)
        plt.plot(epochs_range, acc, label='Training accuracy')
        plt.plot(epochs_range, val_acc, label='Validation accuracy')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(-0.01, 0.6)
        plt.grid(True)
        plt.plot(epochs_range, loss, label='Training loss')
        plt.plot(epochs_range, val_loss, label='Validation loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig('results/train_val_loss.png')


class Evaluator:
    def __init__(self, test_df, model, use_presaved_predictions=False, threshold=0.2):
        """
        - threshold: if the output generated by the corresponding output neuron is greater than threshold -> object detected
        """
        self.model = model
        self.test = self.get_generator_test_set(test_df)
        self.threshold = threshold
        print('Evaluating model on test set and generating metrics...')
        self.evaluate_and_plot(use_presaved_predictions)

    def get_generator_test_set(self, test_df):
        test_generator = ImageDataGenerator(rescale=1/255) 
        return test_generator.flow_from_dataframe(dataframe=test_df,        directory='/scratch/WIT_Dataset/images',
                                                  x_col='image_path',       y_col='taxo_labels', 
                                                  class_mode='categorical', validate_filenames=True,
                                                  target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))

    def evaluate_and_plot(self, use_presaved_predictions):
        # Predictions
        if not use_presaved_predictions:
            print('Predicting probabilities for test set...')
            predictions = self.model.predict(self.test, verbose=1)
            with open('checkpoints/predictions', 'wb') as f:
                np.save(f, predictions)
        else:
            print('Using pre-saved predictions...')
            with open('checkpoints/predictions', 'rb') as f:
                predictions = np.load(f)   
        y_pred = 1 * (predictions > self.threshold)
        y_true = np.zeros(y_pred.shape)
        for row_idx, row in enumerate(self.test.classes):
            for idx in row:
                y_true[row_idx, idx] = 1

        n_classes = y_true.shape[1]
        metrics_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=list(self.test.class_indices), output_dict=True)).transpose()
        metrics_df['index'] = np.concatenate((np.arange(start=0, stop=n_classes), [None, None, None, None]))
        
        # Output macro and micro accuracies
        metrics_df.to_json('results/metrics.json')
        print(metrics_df.tail(4))

        # Average precision score
        # The average_precision_score function computes the average precision (AP) from prediction scores. 
        # The value is between 0 and 1 and higher is better. With random predictions, the AP is the fraction 
        # of positive samples.
        print('\nAverage precision scores (macro and weighted):')
        print(average_precision_score(y_true, y_pred, average='macro'))
        print(average_precision_score(y_true, y_pred, average='weighted'))

        # ROC AUC score
        print('\n ROC AUC score:')
        print(roc_auc_score(y_true, y_pred))

        # Precision and recall for each class
        fig, axs = plt.subplots(1, 2, figsize=(12,12))

        # Precision
        sorted_indices_precision = np.argsort(metrics_df.precision[0:n_classes])
        sorted_precisions_per_class = metrics_df.precision[0:n_classes][sorted_indices_precision]
        # Recall
        sorted_indices_recall = np.argsort(metrics_df.recall[0:n_classes])
        sorted_recalls_per_class = metrics_df.recall[0:n_classes][sorted_indices_recall]
        # Plot!
        axs[0].set_title('Precision per class')
        axs[0].barh(range(y_true.shape[1]), sorted_precisions_per_class, color='blue', alpha=0.6)
        axs[0].set_yticks(range(n_classes))
        axs[0].set_yticklabels(np.array(list(self.test.class_indices.keys()))[sorted_indices_precision])
        axs[0].set_xlabel('Precision')
        axs[0].grid(True)
        axs[1].set_title('Recall per class')
        axs[1].barh(range(y_true.shape[1]), sorted_recalls_per_class, color='blue', alpha=0.6)
        axs[1].set_yticks(range(n_classes))
        axs[1].set_yticklabels([])
        axs[1].set_xlabel('Recall')
        axs[1].grid(True)
        plt.savefig('results/precision_recall.png')
        

def main():

    use_presaved_model = False
    use_presaved_predictions = False

    # ----- Hyperparameters -----
    # Data
    wit_segments = 'one' # 'all'
    min_nr_images_per_class = 1e4
    # Model training
    threshold = 0.2
    epochs = 50
    oversampling = False
    use_class_weights = False
    # ---------------------------

    if not use_presaved_model:
        # Load image<->label dataframe and clean it
        print('\n====================== DATA LOADER ======================\n')
        loader = DataLoader('data/image_labels.json.bz2', min_nr_images_per_class, 64, plot_distribution=True)
        
        # Separate into train/val/test
        print('\n==================== DATA SEPARATOR ======================\n')
        # separator = DataSeparator(loader.data, oversampling=True)
        # test = separator.test
        # test.to_json('data/test_df.json.bz2', compression='bz2')

        # Construct and train mode, plotting accuracy and loss on train & validation sets
        print('\n===================== MODEL TRAINER =====================\n')
        model_trainer = ModelTrainer(separator.train, separator.val, epochs)
        model = model_trainer.model
        model.save('saved_model/my_model')
    else:
        print('\n============ Using pre-saved test-set and saved model ============\n')
        test = pd.read_json('data/test_df.json.bz2', compression='bz2')
        model = tf.keras.models.load_model('saved_model/my_model')
    
    # Evaluate on test set and ouput metrics (precision, recall, accuracy)
    print('\n======================= EVALUATOR =======================\n')
    evaluator = Evaluator(test, model, use_presaved_predictions, threshold)

    # Save results in folder for further analysis
    results_path = os.getcwd() + '/results/' + datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    os.mkdir(results_path)
    with open(results_path + 'result.txt', 'w') as file:
        l1 = f'DATA: \n - WIT segments: {wit_segments}, with {loader.data.shape[0]} images; \
                     \n - train (81%): {separator.train_df.shape[0]}; valid (10%): {separator.val_df.shape[0]}; test (9%): {separator.test_df.shape[0]}                  \
                     \n - distribution of images per class: see plot in {results_path}/class_distribution.png \
                     \n - imbalance level: ?'
        l2 = f'MODEL:\n - epochs: {epochs} \
                     \n - learning rate: {0} \
                     \n - threshold: {threshold} \
                     \n - training loss and accuracy over time: see plot in {results_path}/train_val_loss.png \
                     \n - model summary: ' # to put model summary into print: https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
        l3 = f'HYPER-PARAMETERS: \
                     \n - threshold: {threshold} \
                     \n - epochs: {epochs} \
                     \n - minimal number of images per class: {min_nr_images_per_class} \
                     \n - oversampling: {oversampling} \
                     \n - using class weights: {use_class_weights}'
        file.writelines([l3, l2, l1])


if __name__ == '__main__':
    start = time.time()
    print('Starting classification pipeline of WIT dataset!')
    main()
    end = time.time()
    print(f'Elapsed time: {(end - start) / 60 :.2f} minutes.')