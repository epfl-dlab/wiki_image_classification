import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import stanza
import pickle
import argparse
from headParsing import find_head
from queryLabel import Taxonomy
from tqdm import tqdm
tqdm.pandas()
from utilities import printt

GRAPH_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-category-graph.pkl.bz2'
HEADS_PATH = '/scratch/WikipediaImagesTaxonomy/heads/'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--c', help='initial chunk')
    args = parser.parse_args()

    starting_chunk = int(args.c) if args.c else 0
    printt('Starting from chunk', starting_chunk)

    printt('Loading taxonomy...')
    taxonomy = Taxonomy()
    taxonomy.load_graph(GRAPH_PATH)
    categories = list(taxonomy.G.nodes)

    n_chunks = 20
    batch_size = 64
    categories_chunked = np.array_split(categories, n_chunks)

    find_head('ready')

    for chunk in range(starting_chunk, n_chunks):
        printt(f'Processing chunk {chunk}')
        categories_batched = np.array_split(categories_chunked[chunk], len(categories_chunked[chunk])//batch_size + 1)
        heads = pd.Series(categories_batched).progress_apply(lambda cs: find_head(cs)).explode().values
        
        printt('Saving file...')
        heads_dict = dict(zip(categories, heads))

        with open(HEADS_PATH + f'chunk{chunk}.pkl', 'wb') as fp:
            pickle.dump(heads_dict, fp)
