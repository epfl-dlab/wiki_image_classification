import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import sys

from bert_serving.client import BertClient

sys.path.append("./")
sys.path.append("../../../")

from src.config import *
from src.taxonomy.heuristics import Heuristics
from src.utilities import printt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", help="initial chunk")
    parser.add_argument("-e", "--end", help="final chunk")
    parser.add_argument("-cuda", "--cuda", help="cuda device")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    printt("Loading taxonomy...")
    heuristics = Heuristics()
    heuristics.load_graph(GRAPH_PATH)
    categories = list(heuristics.G.nodes)

    n_chunks = 100
    batch_size = 64
    categories_chunked = np.array_split(categories, n_chunks)
    starting_chunk = int(args.start) if args.start else 0
    end_chunk = int(args.end) if args.end else n_chunks
    printt("Processing from chunk", starting_chunk, "to chunk", end_chunk)

    bc = BertClient()

    for chunk in range(starting_chunk, end_chunk):
        printt(f"Processing chunk {chunk}")
        categories_batched = np.array_split(
            categories_chunked[chunk], len(categories_chunked[chunk]) // batch_size + 1
        )

        embeddings = (
            pd.Series(categories_batched)
            .progress_apply(lambda cs: bc.encode(cs.tolist()))
            .explode()
            .values
        )

        printt("Saving file...")
        embeddings_dict = dict(zip(categories_chunked[chunk], embeddings))

        with open(EMBEDDINGS_PATH + f"chunk{chunk}.pkl", "wb") as fp:
            pickle.dump(embeddings_dict, fp)
