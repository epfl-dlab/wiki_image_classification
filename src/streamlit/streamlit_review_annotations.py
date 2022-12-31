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
    st.markdown(
        f'<img src="{url}" alt="drawing" width="600"/>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    for name in st.session_state.names:
        markdown_str = (
            f"Labels {name.capitalize()}: ["
            + ", ".join(
                [
                    f'<span style="color:{["red", "green"][int(label in file.common_labels)]}">{label}</span>'
                    for label in file[f"labels_{name}"]
                ]
            )
            + "]"
        )

        if file[f"other_text_{name}"]:
            markdown_str += f" ------ Other text: {file[f'other_text_{name}']}"

        st.markdown(markdown_str, unsafe_allow_html=True)


def load_dataset():
    if st.session_state.dataset not in server_state:
        with st.spinner("Loading files..."):
            with server_state_lock[st.session_state.dataset]:
                server_state[st.session_state.dataset] = pd.read_json(
                    GTRUTH_PATH + "annotated/" + st.session_state.dataset
                )

    st.session_state.filesize = len(server_state[st.session_state.dataset])
    names = set()
    for column in filter(
        lambda x: x.startswith("labels_"),
        server_state[st.session_state.dataset].columns,
    ):
        names.add(column.split("_")[-1])
    st.session_state.names = list(names)

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
                lambda f: f.endswith("_combined.json"),
                os.listdir(GTRUTH_PATH + "annotated/"),
            )
        )[0]
        load_dataset()
        st.write("")

    with st.sidebar:
        st.selectbox(
            "Dataset",
            options=list(
                filter(
                    lambda f: f.endswith("_combined.json"),
                    os.listdir(GTRUTH_PATH + "annotated/"),
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
