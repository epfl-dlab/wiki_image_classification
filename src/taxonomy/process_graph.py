import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
tqdm.pandas()

import sys
sys.path.append("../../")

from src.taxonomy.taxonomy import Taxonomy
from src.utilities import printt#, init_logger


HGRAPH_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wheads.pkl.bz2'
# LGRAPH_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wlabels_heuristics_v1.0.pkl.bz2'
LGRAPH_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wlabels_heuristics_simple_v1.0.pkl.bz2'
LGRAPH_H_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-clean-graph-wlabels_heuristics_simple_v1.0.pkl.bz2'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
