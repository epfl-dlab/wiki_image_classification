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
    seed = int(args.seed) if args.seed else 42
    pilot = True

    if pilot:
        name = f"{n}_{seed}_uniform_sample"
    else:
        name = f"{n}_{seed}_{TAXONOMY_VERSION}_{HEURISTICS_VERSION}_balanced_sample"

    files_sample = pd.read_csv(MTURK_PATH + name + "_plain.csv")
    annotated_files = pd.read_csv(MTURK_PATH + name + "_annotated.csv")

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
            dtype=object, columns=["HITId", "WorkerId", "labels", "labels_enriched"]
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
                # labels.remove("None")
                labels_enriched.remove("None")
            if len(labels) == 0:
                warnings.warn(
                    f"No labels selected. HIT {row.HITId}, worker {row.WorkerId}, url {row[f'Input.url{i}']}."
                )
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
                x[["WorkerId", "labels", "labels_enriched"]].values.reshape([-1])
            )
        )
    )
    labels.columns = [
        col
        for lis in [
            [f"WorkerId{i}", f"labels{i}", f"labels_enriched{i}"]
            for i in range(n_assignments)
        ]
        for col in lis
    ]
    labels = labels.reset_index().rename({"index": "url"}, axis=1)

    labels = labels.merge(files_sample[["title", "id", "url"]], on="url", how="left")
    labels = labels.sort_values("HITId").reset_index(drop=True)
    labels = labels[
        [
            "title",
            "id",
            "url",
            "HITId",
            "WorkerId0",
            "WorkerId1",
            "WorkerId2",
            "labels0",
            "labels1",
            "labels2",
            "labels_enriched0",
            "labels_enriched1",
            "labels_enriched2",
        ]
    ]

    def majority_vote(row):
        counter = Counter()
        for i in range(n_assignments):
            counter.update(row[f"labels_enriched{i}"])

        labels = set()
        for label, count in counter.most_common():
            if count >= (n_assignments + 1) // 2:
                labels.add(label)
            else:
                break
        return labels

    labels["labels_enriched_majority"] = labels.apply(majority_vote, axis=1)

    # Descriptive statistics
    print(f"Sample size: {n}, seed {seed}")
    print(
        f"Number of HITs: {annotated_files.HITId.nunique()}, with {n_assignments} assignments per HIT."
    )
    print(f"Number of unique workers: {annotated_files.WorkerId.nunique()}")
    print(annotated_files.WorkerId.value_counts())
    print("\n\nLabel counts:")
    counts = dict(
        Counter(labels.labels_enriched_majority.apply(list).sum()).most_common()
    )
    for label in taxonomy.get_all_labels():
        if label not in counts:
            counts[label] = 0
    counts = pd.DataFrame.from_dict(counts, orient="index", columns=["count"])
    counts = counts.sort_values("count", ascending=False)
    print(counts)

    printt("Saving labels...")
    labels.to_csv(MTURK_PATH + name + "_aggregated.csv", index=False)
    printt("Done.")
