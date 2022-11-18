import argparse
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append("./")
sys.path.append("../../")

from src.config import *
from src.taxonomy.heuristics import Heuristics
from src.utilities import init_logger, printt

logger = init_logger(STREAMLIT_PATH + STREAMLIT_LOG_FILE, logger_name="taxonomy")
logfile = open(STREAMLIT_PATH + STREAMLIT_LOG_FILE, "w+")


def iterativeSampling(
    files,
    images_per_class=50,
    min_images=500,
    mean_noise=0.5,
    var_noise=1,
    random_state=42,
    verbose=1,
):
    """
    Iterative sampling of images, to construct a balanced dataset.
    At each iteration, the class(label) with less predicted images is filled,
    until the balanced set has images_per_class samples of that class.
    To fill a class, the examples with lowest cost are selected, where the cost of each
    image is proportional to the number of images with its labels in the dataset
    and the number of images with its labels already selected in the balanced dataset.
    In addition, a random noise is added to the cost, to avoid biasing the balanced
    dataset by only selecting images with a single label.

    Parameters
    ----------
    files : pd.DataFrame
        Files dataset.
    images_per_class : int, default=50
        Number of desired images per class in the balanced dataset.
    min_images : int, default=500
        Minimum number of images class. Labels with less than
        min_images predicted samples will be discarded.
    mean_noise : float, default=0.5
        Mean of the noise to be added to the cost of each image.
    var_noise : float, default=1
        Variance of the noise to be added to the cost of each image.
    random_state : int, default=42
        Random state for reproducibility.
    verbose : bool, default=1
        Verbosity level.

    Returns
    ----------
    pd.DataFrame
        Balanced dataset.
    """
    assert (
        images_per_class <= min_images
    ), "images_per_class must be less than min_images"

    np.random.seed(random_state)

    encoder = MultiLabelBinarizer()
    y = encoder.fit_transform(files["labels_pred"].apply(list))
    ifiles = pd.DataFrame(y, columns=encoder.classes_)

    index_to_add = []
    count_per_class = ifiles.sum(axis=0)
    count_per_class = count_per_class.sort_values()
    count_per_class = count_per_class[count_per_class >= min_images]
    n_classes = len(count_per_class)
    ifiles = ifiles[count_per_class.index]

    balanced_count_per_class = pd.Series({ind: 0 for ind in count_per_class.index})

    for label, _ in tqdm(count_per_class.iteritems(), total=n_classes):
        n_remaining = images_per_class - balanced_count_per_class[label]
        if n_remaining <= 0:
            continue
        # Update class counts and cost per class
        curr_count_per_class = count_per_class - balanced_count_per_class
        cost_per_class = curr_count_per_class / curr_count_per_class.sum()
        cost_per_class += balanced_count_per_class / images_per_class

        # Get index of images to add to complete the current class
        cost_per_image = ifiles[ifiles[label].astype(bool)] @ cost_per_class
        cost_per_image += np.random.normal(mean_noise, var_noise, len(cost_per_image))
        curr_index_to_add = cost_per_image.sort_values().iloc[:n_remaining].index

        # Update balanced counts and file set
        index_to_add += curr_index_to_add.tolist()
        balanced_count_per_class += ifiles.loc[curr_index_to_add].sum(axis=0)
        ifiles = ifiles.drop(curr_index_to_add)

    balanced_files = files.loc[index_to_add]

    # Sample statistics
    if verbose:
        print("Distribution after sampling:", balanced_count_per_class.value_counts())
        print(
            f"Number of classes with {images_per_class} images = {(balanced_count_per_class == images_per_class).sum()}"
        )
        print(f"Number of images: {len(balanced_files)}")
        print(
            f"Ratio of images with more than one label (original): {(files.labels_pred.apply(len) > 1).sum() / len(files)}"
        )
        print(
            f"Ratio of images with more than one label (balanced): {(balanced_files.labels_pred.apply(len) > 1).sum() / len(balanced_files)}"
        )

    return balanced_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n", help="size of the sample")
    parser.add_argument("-s", "--seed", help="random seed")
    parser.add_argument("-H ", "--how", help="querying scheme")
    args = parser.parse_args()
    n = int(args.n) if args.n else 1000
    seed = int(args.seed) if args.seed else 42
    how = args.how if args.how else HEURISTICS_VERSION

    printt("Reading files...")
    files = pd.read_parquet(FILES_PATH)

    heuristics = Heuristics()
    printt("Loading graph...")
    heuristics.load_graph(EH_GRAPH_PATH)
    printt("Loading mapping...")
    heuristics.set_taxonomy(taxonomy_version=TAXONOMY_VERSION)
    heuristics.set_heuristics(heuristics_version=how)
    printt("Loading done.")

    files["labels_pred"] = files.progress_apply(
        lambda x: heuristics.queryFile(x, debug=False, logfile=logfile),
        axis=1,
        result_type="expand",
    )[0]

    # Create a balanced sample
    files_sample = iterativeSampling(
        files,
        images_per_class=50,
        min_images=500,
        mean_noise=0.5,
        var_noise=0.5,
        random_state=42,
        verbose=1,
    )

    files_sample["labels_true"] = [
        {label: None for label in heuristics.taxonomy.get_all_labels()}
        for _ in range(len(files_sample))
    ]
    files_sample[["labels_pred", "log"]] = files_sample.progress_apply(
        lambda x: heuristics.queryFile(x, debug=True, logfile=logfile),
        axis=1,
        result_type="expand",
    )
    # Dict storing evaluations
    printt("Saving file..")
    files_sample["labels_pred"] = files_sample.apply(
        lambda x: {label: None for label in x.labels_pred}, axis=1
    )
    files_sample.to_json(STREAMLIT_PATH + f"files_{seed}_{n}_{how}.json.bz2")
