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

def align_head(categories, heads):
    '''
    Align category to its head.
    Since stanza 1.4. does not support batch processing, the official advice is to concatenate documents
    with "\n\n". However, this does not respect the boundaries of the original documents, producing potentially more sentences.
    The function matches categories with their heads, by picking the first head contained in the category.
    '''
    category_to_head = {}
    i_head = 0
    for i in range(len(categories)):
        category_to_head[categories[i]] = heads[i_head]
        i_head += 1

        if(i_head == len(heads)):
            break

        while(heads[i_head].lower() in categories[i].lower() and heads[i_head].lower() not in categories[i+1].lower()):
            i_head += 1
    return category_to_head


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
        
        printt('Matching head to category...')
        heads_dict = align_head(categories_chunked[chunk], heads)
        printt('Saving file...')

        with open(HEADS_PATH + f'chunk{chunk}.pkl', 'wb') as fp:
            pickle.dump(heads_dict, fp)
