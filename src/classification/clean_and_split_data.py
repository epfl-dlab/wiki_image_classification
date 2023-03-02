import pandas as pd
import urllib.parse
import os
from tqdm import tqdm
import help_functions as hf
import PIL
from PIL import Image
from sklearn.model_selection import train_test_split
import time

tqdm.pandas()
start = time.time()

hierarchical = True

if hierarchical:
    print('Loading hierarchical labels')
    # Input files
    HEURISTICS_LABELS_PATH = 'data/commonswiki-221218-files-hierarchical-labels.json.bz2'
    # Output files
    HEURISTICS_LABELS_CAN_OPEN_PATH = 'data/commonswiki-230226-files-hierarchical-labels-can-be-opened.json.bz2'
else:
    raise ValueError('No other label options')

labels = pd.read_json(HEURISTICS_LABELS_PATH)

# ------------- Eliminate files with encoding problems

# With this encoded url, only 190k images aren't found, while with "url" 790k aren't found
labels['url'] = labels['url'].progress_apply(lambda encoded_filename : urllib.parse.unquote(encoded_filename).encode().decode('unicode-escape'))
print('Done changing encoding')
labels['can_be_opened'] = labels['url'].progress_apply(lambda url : os.path.isfile('/scratch/WIT_Dataset/images/' + url))
print(f'Total number of images: {labels.shape[0]}.')

labels = labels.loc[labels.can_be_opened == True].reset_index(drop=True)

print(labels.shape)


# ------------- Extra cleaning
print('Starting extra cleaning...')

for index, row in tqdm(labels.iterrows(), total=labels.shape[0]):
    try:
        img = Image.open('/scratch/WIT_Dataset/images/' + row.url)
    except PIL.UnidentifiedImageError:
        print(row.url)
        labels.at[index, 'can_be_opened'] = False
    except Exception as e:
        print(e)
        labels.at[index, 'can_be_opened'] = False
print('Finished extra cleaning! Saving dataframe.')
openable_labels = labels.loc[labels.can_be_opened == True]

openable_labels.to_json(HEURISTICS_LABELS_CAN_OPEN_PATH)
print('Saved!')
hf.print_time(start)