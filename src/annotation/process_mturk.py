import argparse
import sys
import warnings
from collections import Counter

import pandas as pd

sys.path.append("./")
sys.path.append("../../")

from src.config import *
from src.taxonomy.taxonomy import Taxonomy
from src.utilities import printt

UPLOAD_URL = "https://upload.wikimedia.org/wikipedia/commons/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n", help="size of the sample")
    parser.add_argument("-s", "--seed", help="random seed")
    args = parser.parse_args()
    n = int(args.n) if args.n else 1000
    seed = int(args.seed) if args.seed else 0

    printt("Reading files...")
    files = pd.read_parquet(FILES_PATH)
    files_sample = files.sample(n=n, random_state=seed)
    files_sample["url"] = files_sample.url.apply(lambda x: UPLOAD_URL + x)
    annotated_files = pd.read_csv(MTURK_PATH + f"{n}_{seed}_annotated.csv")
    printt("Reading done.")

    taxonomy = Taxonomy()
    taxonomy.set_taxonomy(TAXONOMY_VERSION)
    label_mapping = taxonomy.get_label_mapping()

    cols_labels = list(
        filter(lambda x: x.startswith("Answer"), annotated_files.columns)
    )
    batch_size = 10
    n_assignments = 3

    def extract_labels(row):
        df = pd.DataFrame(
            dtype=object, columns=["HITId", "WorkerId", "Labels", "Labels_enriched"]
        )
        for i in range(batch_size):
            labels = set()
            labels_enriched = set()
            for col in filter(lambda x: x.endswith(str(i)), cols_labels):
                if row[col]:
                    label = col.split(".")[-1][:-1]
                    label = label.replace("_", " & ").title()
                    labels.update([label])
                    if label != "None":
                        labels_enriched.update([label] + label_mapping[label].ancestors)
                    else:
                        labels_enriched.update([label])
            if "None" in labels:
                if len(labels) > 1:
                    warnings.warn(
                        f"Selected None of the above with additional labels. HIT {row.HITId}, worker {row.WorkerId}, url {row[f'Input.url{i}']}."
                    )
                labels.remove("None")
                labels_enriched.remove("None")
            df.loc[row[f"Input.url{i}"]] = [
                row["HITId"],
                row["WorkerId"],
                labels,
                labels_enriched,
            ]
        return df

    printt("Extracting labels...")
    labels = annotated_files.apply(extract_labels, axis=1)
    labels = pd.concat(labels.values)
    labels = (
        labels.reset_index()
        .groupby(["index", "HITId"])
        .apply(
            lambda x: pd.Series(
                x[["WorkerId", "Labels", "Labels_enriched"]].values.reshape([-1])
            )
        )
    )
    labels.columns = [
        col
        for lis in [
            [f"WorkerId{i}", f"Labels{i}", f"Labels_enriched{i}"]
            for i in range(n_assignments)
        ]
        for col in lis
    ]
    labels = labels.reset_index().rename({"index": "url"}, axis=1)

    labels = labels.merge(files_sample[["id", "url"]], on="url")
    labels = labels[
        [
            "id",
            "url",
            "HITId",
            "WorkerId0",
            "WorkerId1",
            "WorkerId2",
            "Labels0",
            "Labels1",
            "Labels2",
            "Labels_enriched0",
            "Labels_enriched1",
            "Labels_enriched2",
        ]
    ]

    def majority_vote(row):
        counter = Counter()
        for i in range(n_assignments):
            counter.update(row[f"Labels_enriched{i}"])

        labels = set()
        for label, count in counter.most_common():
            if count >= (n_assignments + 1) // 2:
                labels.add(label)
            else:
                break
        return labels

    labels["Labels_enriched_majority"] = labels.apply(majority_vote, axis=1)

    printt("Saving labels...")
    labels.to_csv(MTURK_PATH + f"{n}_{seed}_aggregated.csv", index=False)
    printt("Done.")
