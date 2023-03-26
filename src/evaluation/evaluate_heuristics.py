"""
This script evaluates the heuristics based on manually annotated images from MTurk,
selecting the best set of heuristics and tuning their hyperparameters.

Usage:
    python evaluate_heuristics.py
"""
import argparse
import os
import re
import sys
from ast import literal_eval

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


def load_annotations():
    """
    Loads the manually annotated images from MTurk.

    Returns
    ----------
    pd.DataFrame
        The manually annotated images.
    """
    files = pd.read_parquet(FILES_PATH)
    files_annotated = []
    for file in os.listdir(MTURK_PATH):
        if file.endswith("aggregated.csv"):
            df = pd.read_csv(os.path.join(MTURK_PATH, file))
            files_annotated.append(df)
    files_annotated = pd.concat(files_annotated)
    files_annotated = files_annotated.merge(
        files[["id", "categories"]], on=["id"], how="inner"
    )
    files_annotated = files_annotated[
        ["title", "id", "url", "categories", "labels_enriched_majority"]
    ]
    files_annotated = files_annotated.rename(
        {"labels_enriched_majority": "labels"}, axis=1
    )
    files_annotated.labels = files_annotated.labels.apply(literal_eval).apply(list)
    return files_annotated


def main():
    printt("Loading files...")
    files_annotated = load_annotations()

    heuristics = Heuristics()
    printt("Loading graph...")
    heuristics.load_graph(EH_GRAPH_PATH)
    printt("Loading mapping...")
    heuristics.set_taxonomy(taxonomy_version=TAXONOMY_VERSION)

    encoder = MultiLabelBinarizer()
    labels_true = encoder.fit_transform(files_annotated.labels)

    heuristics_list = ["embedding15+headJ", "head+lookahead+depth"]
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
            labels_true, labels_pred, target_names=encoder.classes_, output_dict=True
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

    df_heuristics.to_csv(os.path.join(EVALUATION_PATH, "heuristics_evaluation.csv"))
