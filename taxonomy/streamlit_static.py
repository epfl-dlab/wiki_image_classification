import numpy as np
import pandas as pd
import streamlit as st
import os

SAMPLE_PATH = '/taxonomy/streamlit_data/'
COMMONS_URL = 'https://commons.wikimedia.org/wiki/File:'
UPLOAD_URL = 'https://upload.wikimedia.org/wikipedia/commons/'


def showFile():
    '''
    Show the current file, printing the labels and the log.
    '''

    file = st.session_state.files.iloc[st.session_state.fileList[st.session_state.counter]]

    commons_link = COMMONS_URL + file.url.split('/')[-1]
    image_link = UPLOAD_URL + file.url

    # Display the image with a link to commons
    st.subheader(file.title)
    st.markdown(f'<a href="{commons_link}"><img src="{image_link}"' +
                 ' alt="drawing" width="600"/>', unsafe_allow_html=True)

    st.subheader('Labels: [' + ', '.join(file.labels) + ']')
    
    # Extract the current log
    log_dump = file.log.replace('\n', '  \n ')
    with st.expander("Show log"):
        st.write(log_dump)


def nextFile():
    if(st.session_state.counter == len(st.session_state.fileList) - 1):
        st.session_state.fileList.append(np.random.randint(0, len(st.session_state.files)))

    st.session_state.counter += 1
    st.session_state.previous_disabled = not bool(st.session_state.counter)
    showFile()


def previousFile():
    st.session_state.counter -= 1
    if(not st.session_state.counter):
        st.session_state.previous_disabled = True
    showFile()


def resetSeedNumber():
    np.random.seed(st.session_state.seedNumber)
    st.session_state.fileList = []
    st.session_state.previous_disabled = True
    st.session_state.counter = -1
    nextFile()


def load_dataset():
    with st.spinner('Loading files...'):
        st.session_state.files = pd.read_parquet(SAMPLE_PATH + st.session_state.dataset)
        st.session_state.filesize = len(st.session_state.files)
    resetSeedNumber()


# Initialize session
if 'seedNumber' not in st.session_state:       
    # Other initializations
    st.session_state.seedNumber = 0
    st.session_state.dataset = 'files_0.10.parquet'
    load_dataset()


col1, col2, col3, col4 = st.columns([1, 1, 1.8, 1.8])
col1.text(""); col1.text("");
col1.button('Previous file', key='previousb', on_click=previousFile, disabled=st.session_state.previous_disabled)
col2.text(""); col2.text("");
col2.button('Next file', on_click=nextFile)
col3.selectbox('Dataset', options=os.listdir(SAMPLE_PATH) , key='dataset', on_change=load_dataset)
col4.number_input('Random seed', min_value=0, key='seedNumber', on_change=showFile)
col4.button('Reset', on_click=resetSeedNumber)
