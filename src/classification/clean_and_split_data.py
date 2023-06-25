import sys
import pandas as pd
import urllib.parse
from tqdm import tqdm
import help_functions as hf
from sklearn.model_selection import train_test_split
sys.path.append('../')
from config import FILES_ANNOTATED_PATH
tqdm.pandas()
from PIL import Image 
import PIL
import os
from convert_to_png import convert_to_png
import time
from datetime import date


# Input parameters
split_data_path = f"data/split_data_{date.today().strftime('%d%m%y')}"
os.mkdir(split_data_path)
NR_SAMPLES = 300_000

print('saving results to log file...')

# Save outputs to log file
old_stdout = sys.stdout
log_file = open(split_data_path + '/log.txt', 'w')
sys.stdout = log_file


print('Starting 1...')
# -------------------- 1. Load the dataframe containing the image url and the labels of each image.
start = time.time()
# Input files --> update this to the latest data file generated with Francesco's taxonomy in get_labels.ipynb
labels = pd.read_parquet(FILES_ANNOTATED_PATH).rename(columns={'labels_pred': 'labels'})
labels['url'] = labels['url'].apply(lambda url: '/scratch/WIT_Dataset/images/' + url)


# Remove the data that is used in validation and test of the heuristics.
EVALUATION_PATH = '../../data/evaluation/'
held_out_test = pd.read_parquet(EVALUATION_PATH + "annotated_test.parquet")
held_out_val = pd.read_parquet(EVALUATION_PATH + "annotated_validation.parquet")
held_out_set = pd.concat([held_out_test, held_out_val])
labels = labels[~labels['title'].isin(held_out_set.title.values)]
hf.print_time(start)

# -------------------- 2. Clean

print('Started 2...')
start = time.time()
# ------ 2.1 Fix redirects (dataframes generated in create_redirect_df.ipynb)
print('2.1')
gif_redirects_df = pd.read_json('data/df_of_redirects_gif.json.bz2')
svg_redirects_df = pd.read_json('data/df_of_redirects_svg.json.bz2')

# Merge with redirects_df_gif and redirects_df_svg 
merged_df = labels.merge(gif_redirects_df, left_on='title', right_on='original_title', how='left') \
                  .merge(svg_redirects_df, left_on='title', right_on='original_title', how='left')

# Combine the redirect_url columns
merged_df['redirect_url'] = merged_df['redirect_url_x'].fillna(merged_df['redirect_url_y'])

# Replace the "url" column with "redirect_url" where it is not null
merged_df['url'] = merged_df['redirect_url'].fillna(merged_df['url'])

# Drop the unnecessary columns
merged_df = merged_df.drop(['original_title_x', 'redirect_url_x', 
                            'original_title_y', 'redirect_url_y', 
                            'redirect_title_x', 'redirect_title_y', 
                            'redirect_url'], axis=1)

# ------ 2.2 Eliminate encoding problems and only keep files that can be found
print('2.2')
merged_df['url'] = merged_df['url'].progress_apply(lambda encoded_filename : urllib.parse.unquote(encoded_filename).encode().decode('unicode_escape'))
merged_df['can_be_opened'] = merged_df['url'].progress_apply(lambda url : os.path.isfile(url))
labels_can_be_opened = merged_df.loc[merged_df.can_be_opened == True].reset_index(drop=True)
labels_can_be_opened = labels_can_be_opened.drop(columns=['can_be_opened'])


# ------ 2.3 Remove images whose labels column are empty.
print('2.3')
labels_clean = labels_can_be_opened.loc[labels_can_be_opened['labels'].str.len() != 0].copy()
labels_clean['labels'] = labels_clean.progress_apply(lambda x: list(x.labels), axis=1)
labels_for_conversion = labels_clean.loc[labels_clean['url'].str.lower().str.endswith(('png', 'jpg', 'gif', 'svg'))].copy()

hf.print_time(start)

print('Starting 2.4 ...')
start = time.time()
# ------ 2.4 Convert SVGs and GIFs to PNG in convert_to_png.py and update the url column
print('2.4')
print(f'Shape of labels: {labels_for_conversion.shape}')
labels = convert_to_png(labels=labels_for_conversion)
hf.print_time(start)

# ------ 2.5 Only keep files in Tensorflow's white_list_format
print('2.5')
white_list_formats = ("png", "jpg", "jpeg", "bmp", "ppm", "tif", "tiff")
labels = labels[labels['url'].str.lower().str.endswith(white_list_formats)]

print('Starting 2.6 ...')
start = time.time()
# ------ 2.6 Remove images that give errors when opening with PIL
def try_to_open(url):
    try:
        _ = Image.open(url)
        return True
    except PIL.UnidentifiedImageError:
        print(f'ERROR: UnidentifiedImageError {url}')
    except PIL.Image.DecompressionBombError:
        print(f'ERROR: DecompressionBombError {url}')
    except Exception as e:
        # print(f'ERROR: {e}')
        pass
    return False

labels['can_be_opened'] = labels['url'].progress_apply(lambda x: try_to_open(x))
labels_final = labels[labels['can_be_opened'] == True].drop(columns=['can_be_opened']).reset_index(drop=True)
hf.print_time(start)

labels_final.to_json('data/labels_final.json.bz2')
print('LOG: Success! Saved label_final.')


# -------------------- 3. Sample
# Sample images
print('Starting 3...')
samples = labels_final.sample(n=NR_SAMPLES)

train_df, rest_df = train_test_split(samples, test_size=0.3, random_state=0)
val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=0)
val_df, thresh_df = train_test_split(val_df, test_size=0.5, random_state=0)
print(f'Train set: {train_df.shape[0]} images ({100 * train_df.shape[0] / samples.shape[0]:.2f}%)')
print(f'Test set:  {test_df.shape[0]}  images ({100 * test_df.shape[0] / samples.shape[0]:.2f}%)')
print(f'Val set:  {val_df.shape[0]}  images ({100 * val_df.shape[0] / samples.shape[0]:.2f}%)')
print(f'Thresh set:  {val_df.shape[0]}  images ({100 * thresh_df.shape[0] / samples.shape[0]:.2f}%)')

hf.plot_distribution(dataframe=train_df, filename=split_data_path + '/train_distribution.png')
hf.plot_distribution(dataframe=test_df, filename=split_data_path + '/test_distribution.png')
hf.plot_distribution(dataframe=val_df, filename=split_data_path + '/val_distribution.png')
hf.plot_distribution(dataframe=thresh_df, filename=split_data_path + '/thresh_distribution.png')

train_df.to_json(f'{split_data_path}/train_df.json.bz2', compression='bz2')
test_df.to_json(f'{split_data_path}/test_df.json.bz2', compression='bz2')
val_df.to_json(f'{split_data_path}/val_df.json.bz2', compression='bz2')
thresh_df.to_json(f'{split_data_path}/thresh_df.json.bz2', compression='bz2')