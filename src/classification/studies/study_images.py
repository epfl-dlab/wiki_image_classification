import sys
sys.path.append('..')
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import help_functions as hf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

i = 111
with open('../training_configurations.json', 'r') as fp:
    config = json.load(fp)[str(i)]

test, test_df = hf.get_flow(df_file='../' + config['data_folder'] + '/train_df.json.bz2',
                            batch_size=config['batch_size'],
                            image_dimension=config['image_dimension'])

# Create model
model = hf.create_model(
                        n_labels=len(test.class_indices), 
                        image_dimension=config['image_dimension'], 
                        model_name=config['basemodel'], 
                        number_trainable_layers=config['number_trainable_layers'])
# latest = tf.train.latest_checkpoint('../' + config['results_folder'])
latest = tf.train.latest_checkpoint('../results_thesis/8_flat_model__hierarchical_data')
model.load_weights(latest)

# Predict on test set
print('Predicting on test set:\n')
probs_test = model.predict(test, verbose=1)

y_pred = 1 * (probs_test > 0.5)
np.save('train_y_pred.npy', y_pred, allow_pickle=True, fix_imports=True)


def get_images_from_label(wished_label, img_path, label_names):
    wished_label_idx = label_names.index(wished_label)

    columns = 3
    rows = 10
    _, axs = plt.subplots(rows, columns, figsize=(25, 50))
    indices_with_image = np.argwhere(y_pred[:, wished_label_idx] == 1).flatten()
    random_indices = np.random.choice(indices_with_image, rows*columns, replace=True)

    for i, idx, ax in zip(range(len(random_indices)), random_indices, axs.flatten()):
        path = '/scratch/WIT_Dataset/images/' + test_df.iloc[idx, :].url
        img = load_img(path, target_size=(256, 256))
        # plt.subplot(rows, columns, i + 1)
        ax.imshow(img)
        ax.axis('off')
        gt_str = 'GT: '
        gt_labels = test_df.iloc[idx, :].labels
        gt_labels.sort()
        gt_str += ''.join([label + ', ' for label in gt_labels])
        ax.text(x=0, y=5, s=gt_str[:-2], fontsize=16, bbox=dict(facecolor='white', edgecolor='black'))

        mask = y_pred[idx, :].astype(bool)
        predicted_labels = np.array(label_names)[mask]
        predicted_labels_str = 'Pred: '
        predicted_labels_str += ''.join([label + ', ' for label in predicted_labels])
        ax.text(x=0, y=30, s=predicted_labels_str[:-2], fontsize=16, bbox=dict(facecolor='white', edgecolor='black'))

        plt.suptitle(wished_label, fontsize=50)

        plt.savefig(wished_label + '.png')

label_names = list(test.class_indices.keys())

for label in label_names:
    try:
        print(f'Printing image of label {label}')
        get_images_from_label(label, 'studies/images_pred_and_gt', label_names)
    except Exception as e:
        print(f'Failed displaying predicted images of label {label}')
        print(e)
