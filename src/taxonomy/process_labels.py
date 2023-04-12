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
    heuristics.load_graph(EH_GRAPH_PATH)
    printt("Loading mapping...")
    heuristics.set_taxonomy(TAXONOMY_VERSION)
    heuristics.set_heuristics(HEURISTICS_VERSION)

    printt("Loading files...")
    files = pd.read_parquet(FILES_PATH)

    files["labels_pred"] = files.progress_apply(
        lambda x: heuristics.queryFile(x, debug=False),
        axis=1,
        result_type="expand",
    )[0]

    printt("Saving annotated files...")
    files.to_parquet(FILES_ANNOTATED_PATH)

    nodes = pd.Series(list(heuristics.G.nodes))
    nodes.progress_apply(lambda cat: heuristics.query_category(cat))
    printt("Saving graph...")
    heuristics.dump_graph(LGRAPH_PATH)
    heuristics.dump_graph(LGRAPH_H_PATH, clean=True)
