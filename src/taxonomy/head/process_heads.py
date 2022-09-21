import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import stanza
import pickle
import argparse
from tqdm import tqdm
tqdm.pandas()

import sys
sys.path.append("../../../")

from src.taxonomy.head.headParsing import find_head
from src.taxonomy.taxonomy import Taxonomy
from src.utilities import printt

GRAPH_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-category-graph.pkl.bz2'
HEADS_PATH = '/scratch/WikipediaImagesTaxonomy/heads/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start', help='initial chunk')
    parser.add_argument('-e', '--end', help='final chunk')
    parser.add_argument('-cuda', '--cuda', help='cuda device')
    args = parser.parse_args()

    if(args.cuda):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda    

    printt('Loading taxonomy...')
    taxonomy = Taxonomy()
    taxonomy.load_graph(GRAPH_PATH)
    categories = list(taxonomy.G.nodes)

    n_chunks = 20
    batch_size = 64
    categories_chunked = np.array_split(categories, n_chunks)
    starting_chunk = int(args.start) if args.start else 0
    end_chunk = int(args.end) if args.end else n_chunks
    printt('Processing from chunk', starting_chunk, 'to chunk', end_chunk)

    find_head('ready', use_gpu=bool(args.cuda))

    for chunk in range(starting_chunk, end_chunk):
        printt(f'Processing chunk {chunk}')
        categories_batched = np.array_split(categories_chunked[chunk], len(categories_chunked[chunk])//batch_size + 1)

        heads = pd.Series(categories_batched).progress_apply(lambda cs: find_head(cs, use_gpu=bool(args.cuda))).explode().values
        
        printt('Saving file...')
        heads_dict = dict(zip(categories_chunked[chunk], heads))

        with open(HEADS_PATH + f'chunk{chunk}.pkl', 'wb') as fp:
            pickle.dump(heads_dict, fp)
