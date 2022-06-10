import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
tqdm.pandas()

from queryLabel import Taxonomy
from utilities import printt#, init_logger


HGRAPH_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wheads.pkl.bz2'
# LGRAPH_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wlabels_heuristics_v1.0.pkl.bz2'
LGRAPH_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wlabels_heuristics_simple_v1.0.pkl.bz2'
LGRAPH_H_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-clean-graph-wlabels_heuristics_simple_v1.0.pkl.bz2'

# LOG_PATH = 'process_files.log'
# logger = init_logger(LOG_PATH, logger_name='taxonomy')
# logfile = open(LOG_PATH, 'w+')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--start', help='initial chunk')
    # parser.add_argument('-e', '--end', help='final chunk')
    # parser.add_argument('-cuda', '--cuda', help='cuda device')
    args = parser.parse_args()

    printt('Loading graph...')
    taxonomy = Taxonomy()
    taxonomy.load_graph(HGRAPH_PATH)
    printt('Loading mapping...')
    taxonomy.set_taxonomy(mapping='content_extended')

    nodes = pd.Series(list(taxonomy.G.nodes))
    nodes.progress_apply(lambda cat: taxonomy.get_label(cat, how='heuristics_simple'))
    printt('Saving graph...')
    taxonomy.dump_graph(LGRAPH_PATH)
    taxonomy.dump_graph(LGRAPH_H_PATH, clean=True)
