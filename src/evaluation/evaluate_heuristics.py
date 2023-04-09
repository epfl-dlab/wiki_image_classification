"""
This script evaluates the heuristics based on manually annotated images from MTurk,
selecting the best set of heuristics and tuning their hyperparameters.

Usage:
    python evaluate_heuristics.py
"""
import os
import re
import sys
from ast import literal_eval
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

sys.path.append("./")
sys.path.append("../../")

from src.config import *
from src.taxonomy.heuristics import Heuristics
from src.utilities import printt


def get_heuristics_list(max_n_heuristics=4):
    """
    Construct a list of all possible combinations of heuristics.

    Parameters
    ----------
    max_n_heuristics : int
        The maximum number of heuristics to use.

    Returns
    ----------
    list
        A list of all possible combinations of heuristics.
    """

    base_heuristics = ["head", "depth", "embedding", "look"]
    heuristics_list = []

    for i in range(1, max_n_heuristics + 1):
        heuristics_list += list(permutations(base_heuristics, i))

    heuristics_list = list(map(lambda x: "+".join(x), heuristics_list))

    embeddings_thresholds = np.arange(10, 31, 5, dtype=int)
    new_list = []
    for heuristics in heuristics_list:
        if "head" in heuristics:
            new_list.append(heuristics.replace("head", "headJ"))
    heuristics_list += new_list
    new_list = []
    for heuristics in heuristics_list:
        if "embedding" in heuristics:
            for threshold in embeddings_thresholds:
                new_list.append(
                    heuristics.replace("embedding", f"embedding{threshold}")
                )
    heuristics_list += new_list

    # replace look with lookahead
    heuristics_list = list(
        map(lambda x: x.replace("look", "lookahead"), heuristics_list)
    )

    return heuristics_list


def evaluate_heuristics():
    printt("Loading files...")
    files_annotated = pd.read_parquet(EVALUATION_PATH + "annotated_validation.parquet")
    heuristics = Heuristics()
    printt("Loading graph...")
    heuristics.load_graph(EH_GRAPH_PATH)
    printt("Loading mapping...")
    heuristics.set_taxonomy(taxonomy_version=TAXONOMY_VERSION)

    encoder = MultiLabelBinarizer()
    labels_true = encoder.fit_transform(files_annotated.labels)

    # Evaluate heuristics
    printt("Evaluating heuristics...")
    heuristics_list = get_heuristics_list(max_n_heuristics=4)
    df_list = []
    for heuristics_version in tqdm(heuristics_list):
        heuristics.set_heuristics(heuristics_version=heuristics_version)
        heuristics.reset_labels()
        labels_pred = files_annotated.apply(
            lambda x: heuristics.queryFile(x, debug=False),
            axis=1,
            result_type="expand",
        )[0]
        labels_pred = labels_pred.apply(list)
        labels_pred = encoder.transform(labels_pred)
        report = classification_report(
            labels_true,
            labels_pred,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        )

        df = pd.DataFrame(report).T
        df = df.reset_index().rename({"index": "class"}, axis=1)
        df["heuristics_version"] = heuristics_version
        df_list.append(df)

    df_heuristics = pd.concat(df_list)
    df_heuristics = df_heuristics.pivot_table(
        index=["heuristics_version"], columns=["class"]
    )
    cols = [
        re.sub("[- ]", "_", f"{col[1].lower()}_{col[0]}")
        for col in df_heuristics.columns.values
    ]
    df_heuristics.columns = cols

    printt("Saving results...")
    df_heuristics.to_csv(os.path.join(EVALUATION_PATH, "heuristics_evaluation.csv"))
    printt("Done.")


if __name__ == "__main__":
    evaluate_heuristics()
