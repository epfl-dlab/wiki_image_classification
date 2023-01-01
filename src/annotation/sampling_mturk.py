import argparse
import sys

import pandas as pd

sys.path.append("./")
sys.path.append("../../")

from src.config import *
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
    printt("Reading done.")

    files_sample = files.sample(n=n, random_state=seed)
    files_sample["url"] = files_sample.url.apply(lambda x: UPLOAD_URL + x)
    files_sample = files_sample[["id", "url"]]

    batch_size = 10
    url_batched = files_sample["url"].values.reshape((n // batch_size, batch_size))
    files_sample_reshaped = pd.DataFrame(url_batched)
    files_sample_reshaped.columns = [f"url{i}" for i in range(batch_size)]

    files_sample_reshaped.to_csv(MTURK_PATH + f"{n}_{seed}_sample.csv", index=False)
