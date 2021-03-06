import pandas as pd
import urllib.parse
import os
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from help_functions import get_y_true

print('Started cleaning files!')

MINIMAL_NR_IMAGES = 1_000

NAIVE_LABELS_PATH = 'data/commonswiki-20220220-files-naive-labels.json.bz2'
naive_labels = pd.read_json(NAIVE_LABELS_PATH)

# With this encoded url, only 190k images aren't found, while with "url" 790k aren't found
naive_labels['url'] = naive_labels['url'].apply(lambda encoded_filename : urllib.parse.unquote(encoded_filename).encode().decode('unicode-escape'))
naive_labels['can_be_opened'] = naive_labels['url'].apply(lambda url : os.path.isfile('/scratch/WIT_Dataset/images/' + url))
print(f'Total number of files: {naive_labels.shape[0]}.')

naive_labels = naive_labels.loc[naive_labels.can_be_opened == True].reset_index(drop=True)

naive_labels.rename(columns={'new_labels': 'labels'}, inplace=True)
print(f'Total number of files that can be opened: {naive_labels.shape[0]}.')

import PIL
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

for index, row in tqdm(naive_labels.iterrows(), total=naive_labels.shape[0]):
    try:
        img = Image.open('/scratch/WIT_Dataset/images/' + row.url)
    except PIL.UnidentifiedImageError:
        print(row.url)
        naive_labels.at[index, 'can_be_opened'] = False


# Only keep the files that can be opened
naive_labels = naive_labels.loc[naive_labels.can_be_opened == True].reset_index(drop=True)
naive_labels.to_json('data/commonswiki-20220601-files-naive-labels.json.bz2', compression='bz2')
print(f'ALLELUIA! Saved totally cleaned file to commonswiki-20220601-files-naive-labels.json.bz2')

print(naive_labels.shape)
print(naive_labels.head(2))

_generator = ImageDataGenerator() 
_data = _generator.flow_from_dataframe(dataframe=naive_labels, 
                                       directory='/scratch/WIT_Dataset/images', 
                                       x_col='url', 
                                       y_col='labels', 
                                       class_mode='categorical', 
                                       validate_filenames=False)

y_true = get_y_true(_data.samples, _data.class_indices, _data.classes)

sorted_indices = np.argsort(np.sum(y_true, axis=0))
sorted_images_per_class = y_true.sum(axis=0)[sorted_indices]

mask_kept = y_true.sum(axis=0)[sorted_indices] > MINIMAL_NR_IMAGES
mask_removed = y_true.sum(axis=0)[sorted_indices] < MINIMAL_NR_IMAGES

_ = plt.figure(figsize=(12, 9))
_ = plt.title('Number of times a label is assigned to image (log-scale x-axis)')

_ = plt.barh(np.array(range(y_true.shape[1]))[mask_kept], sorted_images_per_class[mask_kept], color='blue', alpha=0.6)
_ = plt.barh(np.array(range(y_true.shape[1]))[mask_removed], sorted_images_per_class[mask_removed], color='red', alpha=0.6)

_ = plt.yticks(range(y_true.shape[1]), np.array(list(_data.class_indices.keys()))[sorted_indices])
_ = plt.xscale('log')
_ = plt.xlabel('Count')
_ = plt.ylabel('Naive labels')
_ = plt.grid(True)

plt.legend(['Kept', 'Removed'], loc='upper right')

indices_of_classes_to_remove = np.where(np.sum(y_true, axis=0) < MINIMAL_NR_IMAGES)
classes_to_remove = np.array(list(_data.class_indices.keys()))[indices_of_classes_to_remove]

naive_labels['labels'] = naive_labels['labels'].apply(lambda labels: [el for el in labels if el not in classes_to_remove])
naive_labels_clean = naive_labels[naive_labels['labels'].str.len() != 0]
print(f'Removed {naive_labels.shape[0] - naive_labels_clean.shape[0]} images that were only labeled as the removed labels. Left with {naive_labels_clean.shape[0]/1e6:.2f} M images.')

train_df, test_df = train_test_split(naive_labels_clean, test_size=0.05, random_state=17)

print(f'Train set: {train_df.shape[0]} images ({train_df.shape[0] / naive_labels.shape[0]:.2f}%)')
print(f'Test set:  {test_df.shape[0]}  images ({test_df.shape[0] / naive_labels.shape[0]:.2f}%)')

train_df.to_json('data/splitted_dfs_20220601/train_df.json.bz2', compression='bz2')
test_df.to_json('data/splitted_dfs_20220601/test_df.json.bz2', compression='bz2')
print('Saved files!')
