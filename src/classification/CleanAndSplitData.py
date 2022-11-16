import pandas as pd
import urllib.parse
import os
from tqdm import tqdm
import help_functions as hf
import PIL
from PIL import Image
tqdm.pandas()

# INPUT FILES
HEURISTICS_LABELS_PATH = 'data/commonswiki-20221114-files-heuristic-labels.json.bz2'

# OUTPUT FILES
HEURISTICS_LABELS_CAN_OPEN_PATH = 'data/commonswiki-221116-files-hierarchical-labels-can-be-opened.json.bz2'
SPLIT_DATA_PATH = 'data/split_hierarchical_data_221116'

labels = pd.read_json(HEURISTICS_LABELS_PATH)


# ------------- Only keep files that can be opened (eliminate encoding problems)

# With this encoded url, only 190k images aren't found, while with "url" 790k aren't found
labels['url'] = labels['url'].progress_apply(lambda encoded_filename : urllib.parse.unquote(encoded_filename).encode().decode('unicode-escape'))
print('Done changing encoding')
labels['can_be_opened'] = labels['url'].progress_apply(lambda url : os.path.isfile('/scratch/WIT_Dataset/images/' + url))
print(f'Total number of files: {labels.shape[0]}.')

labels = labels.loc[labels.can_be_opened == True].reset_index(drop=True)

print(f'Total number of files that can be opened: {labels.shape[0]}.')
print(labels.shape)



# ------------- Only keep files in jpg format (eliminate problem of png and webp format)

labels_jpg = labels.loc[labels.url.str.endswith(('.jpg', '.JPG', '.Jpg'))]
print(f'{labels_jpg.shape[0]} images of type .jpg')



# ------------- Extra cleaning

for index, row in tqdm(labels_jpg.iterrows(), total=labels_jpg.shape[0]):
    try:
        img = Image.open('/scratch/WIT_Dataset/images/' + row.url)
    except PIL.UnidentifiedImageError:
        print(row.url)
        labels_jpg.at[index, 'can_be_opened'] = False
    except Exception as e:
        print(e)
        labels_jpg.at[index, 'can_be_opened'] = False

openable_labels = labels_jpg.loc[labels_jpg.can_be_opened == True]

openable_labels.to_json(HEURISTICS_LABELS_CAN_OPEN_PATH)