import sys

sys.path.append("./")
sys.path.append("../../")

import argparse
import json
import os
from collections import Counter

import pandas as pd

from src.config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n", help="size of the sample")
    parser.add_argument("-s", "--seed", help="random seed")
    args = parser.parse_args()
    n = int(args.n) if args.n else 100
    seed = int(args.seed) if args.seed else 0

    df_list = []
    path = MTURK_PATH + "annotated/"
    for file in filter(
        lambda x: x.startswith(f"{n}_{seed}_")
        and x.endswith(".json")
        and not "combined" in x,
        os.listdir(path),
    ):
        name = file.split("_")[3].split(".")[0].lower()
        with open(path + file, "r") as f:
            data = pd.DataFrame(json.load(f))
            df_list.append(data[["labels", "other_text"]].add_suffix("_" + name))
    df = pd.concat(df_list, axis=1)
    df.insert(0, "id", data["id"])
    df.insert(1, "url", data["url"])

    # Combine labels
    df_labels = df[list(filter(lambda x: x.startswith("labels_"), df.columns))]
    df_other = df[list(filter(lambda x: x.startswith("other_text_"), df.columns))]
    df["common_labels"] = df_labels.apply(
        lambda x: set.intersection(*[set(l) for l in x.to_list()]), axis=1
    )
    df["all_labels"] = df_labels.apply(
        lambda x: set.union(*[set(l) for l in x.to_list()]), axis=1
    )
    df["all_others"] = df_other.apply(lambda x: [el.lower() for el in x if el], axis=1)
    df["to_review"] = df.apply(
        lambda x: len(x.common_labels) != len(x.all_labels) or len(x.all_others) > 0,
        axis=1,
    )

    # Statistics
    print(f"Out of {len(df)} images, {sum(df.to_review)} need to be reviewed")
    print(f"Other texts: {df.all_others.sum()}")
    print(
        f"Conflict labes: {Counter([x for l in df.apply(lambda x: x.all_labels - x.common_labels, axis=1).values for x in l]).most_common()}\n"
    )
    print(
        f"Common labels: {Counter([x for l in df.common_labels.values for x in l]).most_common()}\n"
    )
    print(
        f"All labels: {Counter([x for l in df.all_labels.values for x in l]).most_common()}"
    )

    # Save combined
    df.to_json(path + f"{n}_{seed}_sample_combined.json", orient="records")
