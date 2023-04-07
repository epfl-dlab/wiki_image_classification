"""
Combine the results of the MTurk evaluation into a single CSV file.

Usage:
    python mturk_combine.py
"""
import os
import sys
from ast import literal_eval
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

sys.path.append("./")
sys.path.append("../../")

from src.config import *
from src.utilities import printt


def print_counts(df):
    print("\n\nLabel counts:")
    counts = dict(Counter(df.labels.apply(list).sum()).most_common())
    counts = pd.DataFrame.from_dict(counts, orient="index", columns=["count"])
    counts = counts.sort_values("count", ascending=False)
    print(counts)


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
    files_annotated.drop_duplicates(subset=["id"])

    ## Map taxonomy v1.3 to v1.4
    files_annotated.labels = files_annotated.labels.apply(
        lambda x: [
            label
            if (label != "Fossils" and label != "Geology")
            else "Geology & Fossils"
            for label in x
            if label != "Belief"
        ]
    )

    print_counts(files_annotated)

    return files_annotated


if __name__ == "__main__":
    files_annotated = load_annotations()

    ## Split into validation (for the heuristics) and test set
    X = files_annotated.index.values[:, np.newaxis]
    encoder = MultiLabelBinarizer()
    y = encoder.fit_transform(files_annotated.labels)

    np.random.seed(42)
    X_val, y_val, X_test, y_test = iterative_train_test_split(X, y, test_size=0.7)

    files_annotated_val = files_annotated.loc[X_val.reshape(-1)]
    files_annotated_test = files_annotated.loc[X_test.reshape(-1)]

    print("\n\nValidation set:")
    print_counts(files_annotated_val)

    print("\n\nTest set:")
    print_counts(files_annotated_test)

    ## Save files
    files_annotated_val.to_csv(
        os.path.join(EVALUATION_PATH, "annotated_validation.csv"), index=False
    )
    files_annotated_test.to_csv(
        os.path.join(EVALUATION_PATH, "annotated_test.csv"), index=False
    )
