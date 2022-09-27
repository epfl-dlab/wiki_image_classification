import os
import sys

import numpy as np
import pandas as pd
from streamlit_server_state import server_state, server_state_lock

import streamlit as st

sys.path.append("./")
sys.path.append("../../")

from src.config import *

COMMONS_URL = "https://commons.wikimedia.org/wiki/File:"
UPLOAD_URL = "https://upload.wikimedia.org/wikipedia/commons/"


def showFile():
    """
    Show the current file, printing the labels and the log.
    """

    file = server_state[st.session_state.dataset].iloc[st.session_state.counter]

    commons_link = COMMONS_URL + file.url.split("/")[-1]
    image_link = UPLOAD_URL + file.url

    # Display the image with a link to commons
    st.subheader(file.title)
    st.markdown(
        f'<a href="{commons_link}"><img src="{image_link}"'
        + ' alt="drawing" width="600"/>',
        unsafe_allow_html=True,
    )

    st.subheader("Labels: [" + ", ".join(file.labels) + "]")

    # Hide first botton of radio widget, so that by default no option is selected
    st.markdown(
        """ <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    index_map = {None: 0, 0: 2, 1: 1}
    for label, value in file.labels.items():
        st.radio(
            label,
            ("-", "Correct", "Wrong"),
            key=file.title + label,
            index=index_map[value],
            on_change=evaluate_label,
            args=(file, label),
        )

    # Extract the current log
    log_dump = file.log.replace("\n", "  \n ")
    with st.expander("Show log"):
        st.write(log_dump)


def evaluate_label(file, label):
    index_map = {"Correct": 1, "Wrong": 0}
    evaluation = index_map[st.session_state[file.title + label]]
    with server_state_lock[st.session_state.dataset]:
        file.labels[label] = evaluation
    showFile()


def load_dataset():
    if st.session_state.dataset not in server_state:
        with st.spinner("Loading files..."):
            with server_state_lock[st.session_state.dataset]:
                server_state[st.session_state.dataset] = pd.read_json(
                    STREAMLIT_PATH + st.session_state.dataset
                )

    st.session_state.filesize = len(server_state[st.session_state.dataset])
    showFile()


def next_unevaluated():
    evaluated = server_state[st.session_state.dataset].labels.apply(
        lambda x: None in x.values()
    )
    evaluated = np.where(evaluated.values[st.session_state.counter + 1 :])[0]
    if evaluated.size:
        st.session_state.counter += evaluated[0] + 1
    else:
        st.warning("All images in this dataset have already been evaluated!")
    showFile()


def main():
    # Initialize session
    if "counter" not in st.session_state:
        # st.set_page_config(layout="wide")
        st.session_state.counter = 0
        st.session_state.filesize = 1
        st.session_state.previous_disabled = True
        st.session_state.dataset = "files_42_10000_heuristics_simple.json.bz2"
        load_dataset()
        st.write("")

    with st.sidebar:
        st.selectbox(
            "Dataset",
            options=list(
                filter(lambda f: f.endswith("bz2"), os.listdir(STREAMLIT_PATH))
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
            "Go to next unevaluated image",
            on_click=next_unevaluated,
            disabled=(st.session_state.counter == st.session_state.filesize - 1),
        )
        # Blank vertical space
        for _ in range(10):
            st.text("")
        # Hard to compress within streamlit, need to do it manually afterwards
        st.download_button(
            "Download",
            data=server_state[st.session_state.dataset].to_json(),
            file_name=st.session_state.dataset.split(".")[0] + "_annotated.json",
            on_click=showFile,
        )


if __name__ == "__main__":
    main()
