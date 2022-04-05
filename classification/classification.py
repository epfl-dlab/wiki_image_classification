# Common libraries
import pandas as pd
import numpy as np
import urllib.parse
import time
from matplotlib import pyplot as plt

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten


# Settings
IMAGE_DIMENSION = 64
EPOCHS = 50


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
            plt.savefig('figures/class_distribution.png')

        return df


class DataSeparator:
    def __init__(self, image_label_df, oversampling=False):

        train_df, test_df = train_test_split(image_label_df, test_size=0.1, random_state=17)
        train_df, val_df = train_test_split(train_df, train_size=0.9, random_state=17)

        # Data generator for training and validation sets
        if oversampling:
            train_generator = ImageDataGenerator(validation_split=0.10,  rescale=1/255,
                                                 rotation_range=40,      width_shift_range=0.2,
                                                 height_shift_range=0.2, shear_range=0.2,
                                                 zoom_range=0.2,         horizontal_flip=True, fill_mode='nearest') 
        else:
            train_generator = ImageDataGenerator(validation_split=0.10, rescale=1/255)

        print('\n----------- Train images -----------')
        self.train = train_generator.flow_from_dataframe(dataframe=train_df,       directory='/scratch/WIT_Dataset/images', 
                                                         x_col='image_path',       y_col='taxo_labels', 
                                                         class_mode='categorical', subset='training',
                                                         validate_filenames=True,  target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION)) # the dimensions to which all images found will be resized.

        print('\n----------- Validation images -----------')          
        self.val = train_generator.flow_from_dataframe(dataframe=val_df,         directory='/scratch/WIT_Dataset/images', 
                                                       x_col='image_path',       y_col='taxo_labels', 
                                                       class_mode='categorical', subset='validation',
                                                       validate_filenames=True,  target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))

        # Data generator for test set
        print('\n----------- Test images -----------')
        # balanced_test_df = self.balance_test(test_df)    
        self.test = test_df     
        # test_generator = ImageDataGenerator(rescale=1/255) 
        # self.test = test_generator.flow_from_dataframe(dataframe=test_df,        directory='/scratch/WIT_Dataset/images',
        #                                                x_col='image_path',       y_col='taxo_labels', 
        #                                                class_mode='categorical', validate_filenames=True,
        #                                                target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))

    def balance_test(self, test_df):

        pass


class ModelTrainer:
    def __init__(self, train, val, use_class_weights=False):
        self.train = train
        self.val = val
        self.n_classes = len(train.class_indices)
        self.model = self.construct_model()
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
        history = self.model.fit(self.train, epochs=EPOCHS, steps_per_epoch=15, 
                                 validation_data=self.val, validation_steps=7,
                                 verbose=2)
        return history
    
    def plot_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(EPOCHS)

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
        plt.savefig('figures/train_val_loss.png')


class Evaluator:
    def __init__(self, test_df, model, threshold=0.1):
        """
        - threshold: if the output generated by the corresponding output neuron is greater than threshold -> object detected
        """
        self.model = model
        self.test = self.get_generator_test_set(test_df)
        self.threshold = threshold
        self.evaluate()

    def get_generator_test_set(self, test_df):
        test_generator = ImageDataGenerator(rescale=1/255) 
        return test_generator.flow_from_dataframe(dataframe=test_df,        directory='/scratch/WIT_Dataset/images',
                                                  x_col='image_path',       y_col='taxo_labels', 
                                                  class_mode='categorical', validate_filenames=True,
                                                  target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))


    def evaluate(self):
        # test_accuracy_score = self.model.evaluate(self.test, verbose=0)
        # print(f'test_accuracy_score: {test_accuracy_score}')
        # print("Accuracy on test set: {:.4f}%".format(test_accuracy_score[1] * 100)) 
        # print("Loss on test set: ", test_accuracy_score[0])

        predictions = self.model.predict(self.test)
        y_pred = 1 * (predictions > self.threshold)

        y_true = np.zeros(y_pred.shape)
        for row_idx, row in enumerate(self.test.classes):
            for idx in row:
                y_true[row_idx, idx] = 1

        metrics_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=list(self.test.class_indices), output_dict=True)).transpose()
        metrics_df['index'] = np.concatenate((np.arange(start=0, stop=6), [None, None, None, None]))
        
        # Output macro and micro accuracies
        ax = metrics_df.tail(4).plot()
        fig = ax.get_figure()
        fig.savefig('figures/metrics_df.png')

        fig, axs = plt.subplots(1, 2, figsize=(18,3))
        axs[0].set_xlabel('Label')
        axs[0].set_ylabel('Precision')
        axs[0].set_title('Precision')
        axs[1].set_xlabel('Label')
        axs[1].set_ylabel('Recall')
        axs[1].set_title('Recall')
        axs[0].set_ylim([-0.05, 1.05])
        axs[1].set_ylim([-0.05, 1.05])
        axs[0].set_xticks(range(y_true.shape[1]), range(y_true.shape[1]))
        axs[1].set_xticks(range(y_true.shape[1]), range(y_true.shape[1]))
        axs[0].plot(range(y_true.shape[1]), metrics_df.precision[0:y_true.shape[1]], 'bo')
        axs[1].plot(range(y_true.shape[1]), metrics_df.recall[0:y_true.shape[1]], 'bo')
        plt.savefig('figures/precision_recall.png')
        

def main():

    use_presaved_model = True

    if not use_presaved_model:
        # Load image<->label dataframe and clean it
        print('\n====================== DATA LOADER ======================\n')
        loader = DataLoader('data/image_labels.json.bz2', 1e4, 64, plot_distribution=True)
        
        # Separate into train/val/test
        print('\n==================== DATA SEPARATOR ======================\n')
        separator = DataSeparator(loader.data, oversampling=False)
        test = separator.test

        # Construct and train mode, plotting accuracy and loss on train & validation sets
        print('\n===================== MODEL TRAINER =====================\n')
        model_trainer = ModelTrainer(separator.train, separator.val)
        model = model_trainer.model
        # model.save('saved_model/my_model')  
    else:
        test = pd.read_json('data/test_df.json.bz2', compression='bz2')
        model = tf.keras.models.load_model('saved_model/my_model')
    
    # Evaluate on test set and ouput metrics (precision, recall, accuracy)
    print('\n======================= EVALUATOR =======================\n')
    evaluator = Evaluator(test, model, use_presaved_model)


if __name__ == '__main__':
    start = time.time()
    print('Starting classification pipeline of WIT dataset!')
    main()
    end = time.time()
    print(f'Elapsed time: {(end - start) / 60 :.2f} minutes.')