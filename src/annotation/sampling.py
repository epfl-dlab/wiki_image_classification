"""
Sample a subset of the image dataset for evaluation, either
uniformly at random or stratified by label (as predicted by 
the "base" heuristics), in a balanced way.

Usage: 
    python sampling.py -n <n> -s <seed>

Other settings might need to be adjusted in the main code.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

tqdm.pandas()
sys.path.append("./")
sys.path.append("../../")

from src.config import *
from src.taxonomy.heuristics import Heuristics
from src.utilities import init_logger, printt

UPLOAD_URL = "https://upload.wikimedia.org/wikipedia/commons/"
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
    weighted=False,
):
    """
    Iterative sampling of images, to construct a balanced dataset.
    At each iteration, the class(label) with less predicted images is filled,
    until the balanced set has images_per_class samples of that class.

    In the unweighted version, classes(labels) are filled by sampling uniformly
    at random <images_per_class> images from each class.

    In the weighted version (DEPRECATED), each image is given a cost proportional to the
    number of images with the same labels in the original dataset and the in-progress
    balanced dataset. Classes(labels) are filled by adding iteratively the image with
    the lowest cost until there are <images_per_class> images, with the addition of a
    random noise to avoid biasing the balanced dataset by only selecting
    images with a single label.

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
    ifiles = ifiles.sample(frac=1, random_state=random_state)

    index_to_add = []
    count_per_class = ifiles.sum(axis=0)
    count_per_class = count_per_class.sort_values()
    count_per_class = count_per_class[count_per_class >= min_images]
    n_classes = len(count_per_class)
    ifiles = ifiles[count_per_class.index]

    balanced_count_per_class = pd.Series({ind: 0 for ind in count_per_class.index})

    for label, _ in tqdm(count_per_class.iteritems(), total=n_classes):
        n_remaining = images_per_class
        if weighted:
            n_remaining -= balanced_count_per_class[label]
        if n_remaining <= 0:
            continue
        # Update class counts and cost per class
        curr_count_per_class = count_per_class - balanced_count_per_class
        if weighted:
            cost_per_class = curr_count_per_class / curr_count_per_class.sum()
            cost_per_class += balanced_count_per_class / images_per_class
        else:
            cost_per_class = np.zeros(len(curr_count_per_class))

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
        # Not sorted on purpose - same order as how the sample is populated
        print("Distribution after sampling:\n", balanced_count_per_class)
        print(
            f"Number of classes with {images_per_class} images = {(balanced_count_per_class == images_per_class).sum()}"
        )
        print(f"Number of images: {len(balanced_files)}")
        print(
            f"Ratio of images with more than one label (original, only among images with labels): {(files.labels_pred.apply(len) > 2).sum() / (files.labels_pred.apply(len) >= 1).sum()}"
        )
        print(
            f"Ratio of images with more than one label (balanced): {(balanced_files.labels_pred.apply(len) > 2).sum() / len(balanced_files)}"
        )

    return balanced_files


def uniform_sampling(files, n, seed):
    """
    Uniform sampling of images.
    """
    printt("Loading done.")
    printt("Sampling...")
    files_sample = files.sample(n=n, random_state=seed)
    return files_sample


def stratified_sampling(files, n, seed, taxonomy_version, heuristics_version):
    """
    Stratified sampling of images, to construct a balanced dataset.
    """
    heuristics = Heuristics()
    printt("Loading graph...")
    heuristics.load_graph(EH_GRAPH_PATH)
    printt("Loading mapping...")
    heuristics.set_taxonomy(taxonomy_version=taxonomy_version)
    heuristics.set_heuristics(heuristics_version=heuristics_version)
    printt("Loading done.")

    printt("Predicting labels...")
    files["labels_pred"] = files.progress_apply(
        lambda x: heuristics.queryFile(x, debug=False, logfile=logfile),
        axis=1,
        result_type="expand",
    )[0]

    printt("Sampling...")
    files_sample = iterativeSampling(
        files,
        images_per_class=n,
        min_images=n,
        mean_noise=0,
        var_noise=0.2,
        random_state=seed,
        verbose=1,
        weighted=False,
    )
    return files_sample


def save_streamlit(files_sample, name):
    """
    Save the dataset for streamlit.
    """
    files_sample_loc = files_sample.copy()
    # Default dictionary for streamlit evaluation
    files_sample_loc["labels_true"] = [
        {label: None for label in heuristics.taxonomy.get_all_labels()}
        for _ in range(len(files_sample_loc))
    ]

    # Predicting labels again with debug=True, to get the logs for the sample
    printt("Resetting labels...")
    heuristics.reset_labels()
    files_sample_loc[["labels_pred", "log"]] = files_sample_loc.progress_apply(
        lambda x: heuristics.queryFile(x, debug=True, logfile=logfile),
        axis=1,
        result_type="expand",
    )
    files_sample_loc["labels_pred"] = files_sample_loc.apply(
        lambda x: {label: None for label in x.labels_pred}, axis=1
    )

    printt("Saving file...")
    files_sample_loc.to_json(STREAMLIT_PATH + name + ".json.bz2")


def save_grounded_truth(files_sample, name):
    """
    Save the dataset for grounded truth evaluation.
    """
    files_sample_loc = files_sample.copy()
    files_sample_loc["url"] = files_sample_loc.url.apply(lambda x: UPLOAD_URL + x)
    files_sample_loc = files_sample_loc[["id", "title", "url"]]
    files_sample_loc.to_json(GTRUTH_PATH + name + ".json.bz2", orient="records")


def save_mturk(files_sample, name, batch_size):
    """
    Save the dataset for MTurk evaluation.
    """
    files_sample_loc = files_sample.copy()
    files_sample_loc["url"] = files_sample_loc.url.apply(lambda x: UPLOAD_URL + x)
    files_sample_loc.to_csv(MTURK_PATH + name + "_plain.csv")

    # Splitting the dataset into batches
    files_sample_loc = files_sample_loc[["id", "url"]]
    url_batched = files_sample_loc["url"].values.reshape((-1, batch_size))
    files_sample_reshaped = pd.DataFrame(url_batched)
    files_sample_reshaped.columns = [f"url{i}" for i in range(batch_size)]

    printt("Saving file...")
    files_sample_reshaped.to_csv(MTURK_PATH + name + "_batched.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n", help="size of the sample")
    parser.add_argument("-seed", "--seed", help="random seed")
    args = parser.parse_args()

    # If uniform sampling (balanced=False), n is the number of total images in the sample
    # If stratified sampling (balanced=True), n is the number of images per class
    n = int(args.n) if args.n else 1000
    seed = int(args.seed) if args.seed else 42

    #############  DEFINE THE APPROPRIATE SETTNIGS #############
    version = TAXONOMY_VERSION
    how = HEURISTICS_VERSION

    ## EARLY EXPERIMENTS
    # balanced = False
    # saving = "streamlit"

    ## GROUNDED TRUTH, i.e. manual annotation to select the taxonomy
    # balanced = False
    # saving = "gtruth"

    ## MTURK PILOT
    # balanced = False
    # saving = "mturk"

    ## MTURK STUDY
    balanced = True
    saving = "mturk"
    ############################################################

    printt("Reading files...")
    files = pd.read_parquet(FILES_PATH)
    printt("Reading done.")

    if balanced:
        files_sample = stratified_sampling(files, n, seed, version, how)
        name = f"{n}_{seed}_{version}_{how}_balanced_sample"
    else:
        files_sample = uniform_sampling(files, n, seed)
        name = f"{n}_{seed}_uniform_sample"

    if saving == "streamlit":
        save_streamlit(files_sample, name)
    elif saving == "gtruth":
        save_grounded_truth(files_sample, name)
    elif saving == "mturk":
        save_mturk(files_sample, name, batch_size=10)
    else:
        raise ValueError("Invalid saving option")

    printt("Done.")
