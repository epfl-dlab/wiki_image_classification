import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
tqdm.pandas()

import sys
sys.path.append("./")
sys.path.append("../../")

from src.taxonomy.taxonomy import Taxonomy
from src.utilities import printt#, init_logger
from src.config import *


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
