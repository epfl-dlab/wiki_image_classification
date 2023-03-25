import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit_nested_layout
from streamlit_server_state import server_state, server_state_lock

import streamlit as st

sys.path.append("./")
sys.path.append("../../")

from src.config import *
from src.taxonomy.taxonomy import Taxonomy

COMMONS_URL = "https://commons.wikimedia.org/wiki/File:"
UPLOAD_URL = "https://upload.wikimedia.org/wikipedia/commons/"


def showFile():
    """
    Show the current file, printing the labels and the log.
    """

    file = server_state[st.session_state.dataset].iloc[st.session_state.counter]

    url = file.url

    # Display the image with a link to commons
    st.subheader(file.title)
    st.markdown(
        f'<a href="{COMMONS_URL + url.split("/")[-1]}"><img src="{url}" alt="drawing" width="600"/>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(f"HIT {file.HITId}")
    for i in range(st.session_state.n_assignments + 1):
        emoji = "✔" if file[f"AssignmentStatus{i}"] == "Approved" else "❓"
        markdown_str = (
            f"{emoji} Labels {file[f'WorkerId{i}']}: ["
            + ", ".join(
                [
                    f'<span style="color:{["red", "green"][int(label in file.labels_enriched_majority)]}">{label}</span>'
                    for label in file[f"labels{i}"]
                ]
            )
            + "]"
        )
        st.markdown(markdown_str, unsafe_allow_html=True)

    st.markdown(f"Labels majority: {file.labels_enriched_majority}")


def load_dataset():
    if st.session_state.dataset not in server_state:
        with st.spinner("Loading files..."):
            df = pd.read_csv(MTURK_PATH + st.session_state.dataset)
            for column in filter(
                lambda x: x.startswith("labels"),
                df.columns,
            ):
                df[column] = df[column].apply(lambda x: eval(x))
            server_state[st.session_state.dataset] = df

    st.session_state.filesize = len(server_state[st.session_state.dataset])
    n_assignments = 0
    for column in filter(
        lambda x: x.startswith("WorkerId"),
        server_state[st.session_state.dataset].columns,
    ):
        n_assignments = max(n_assignments, int(column[-1]))
    st.session_state.n_assignments = n_assignments

    time.sleep(1)
    showFile()


def next_review():
    to_review = server_state[st.session_state.dataset].to_review
    to_review = np.where(to_review.values[st.session_state.counter + 1 :])[0]
    if to_review.size:
        st.session_state.counter += to_review[0] + 1
    else:
        st.warning("All images in this dataset have already been reviewed!")
    showFile()


def main():
    # Initialize session
    if "counter" not in st.session_state:
        # st.set_page_config(layout="wide")
        st.session_state.counter = 0
        st.session_state.show_predictions = True
        st.session_state.filesize = 1
        st.session_state.previous_disabled = True
        st.session_state.dataset = list(
            filter(
                lambda f: f.endswith("_aggregated.csv"),
                os.listdir(MTURK_PATH),
            )
        )[0]
        load_dataset()
        st.write("")

    with st.sidebar:
        st.selectbox(
            "Dataset",
            options=list(
                filter(
                    lambda f: f.endswith("_aggregated.csv"),
                    os.listdir(MTURK_PATH),
                )
            ),
            key="dataset",
            on_change=load_dataset,
        )

        st.number_input(
            "File counter",
            min_value=0,
            max_value=st.session_state.filesize - 1,
            key="counter",
            on_change=showFile,
        )
        st.button(
            "Go to next image to review",
            on_click=next_review,
            disabled=(st.session_state.counter == st.session_state.filesize - 1),
        )


if __name__ == "__main__":
    main()
