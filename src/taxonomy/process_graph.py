import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

import sys

sys.path.append("./")
sys.path.append("../../")

from src.config import *
from src.taxonomy.heuristics import Heuristics
from src.utilities import printt  # , init_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    printt("Loading graph...")
    heuristics = Heuristics()
    heuristics.load_graph(HGRAPH_PATH)
    printt("Loading mapping...")
    heuristics.set_taxonomy(version=TAXONOMY_VERSION)

    nodes = pd.Series(list(heuristics.G.nodes))
    nodes.progress_apply(lambda cat: heuristics.get_label(cat, how=HEURISTICS_VERSION))
    printt("Saving graph...")
    heuristics.dump_graph(LGRAPH_PATH)
    heuristics.dump_graph(LGRAPH_H_PATH, clean=True)
