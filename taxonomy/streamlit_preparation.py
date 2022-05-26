import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm
from utilities import init_logger, printt
from queryLabel import Taxonomy

GRAPH_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-category-graph.pkl.bz2'
FILES_PATH = '/scratch/WikipediaImagesTaxonomy/commonswiki-20220220-files.parquet'
SAMPLE_PATH = 'streamlit_data/'
LOG_PATH = 'streamlit_preparation.log'


logger = init_logger(LOG_PATH, logger_name='taxonomy')
logfile = open(LOG_PATH, 'w+')


def initialize():
    printt('Reading files...')
    files = pd.read_parquet(FILES_PATH)

    taxonomy = Taxonomy()
    printt('Loading graph...')
    taxonomy.load_graph(GRAPH_PATH)
    printt('Loading mapping...')
    taxonomy.set_taxonomy(mapping='content_extended')
    printt('Loading lexical parser...')
    taxonomy.get_head('CommonsRoot')

    return files, taxonomy


def get_sample(n, seed):
    return files.sample(n, random_state=seed)


def queryFile(file, how='heuristics'):
    '''
    Given one file, a row of the files DataFrame, queries recursively all
    the categories and returns the final labels.
    '''

    labels = set()
    for category in file.categories:
        logger.debug(f'Starting search for category {category}')
        cat_labels = taxonomy.get_label(category, how=how)
        logger.debug(f'Ending search for category {category} with resulting labels {cat_labels}')
        logger.debug(f'---------------------------------------------------')
        labels |= cat_labels
    logger.debug(f'Final labels: {labels}')
    log = logfile.read()
    return labels, log



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n', help='size of the sample')
    parser.add_argument('-s', '--seed', help='random seed')
    args = parser.parse_args()

    files, taxonomy = initialize()

    n = int(args.n) if args.n else 1000
    seed = int(args.seed) if args.seed else 0
    files_sample = get_sample(n, seed)
    tqdm.pandas()
    files_sample[['labels', 'log']] = files_sample.progress_apply(lambda x: queryFile(x), 
                                                                  axis=1, result_type="expand")
    # Dict storing evaluations
    files_sample['labels'] = files_sample.apply(lambda x: {label: None for label in x.labels}, axis=1)                                                                  
    files_sample.to_json(SAMPLE_PATH + f'files_{seed}.{n}.json.bz2')
