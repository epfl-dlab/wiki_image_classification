import numpy as np
import pandas as pd
import streamlit as st

from utilities import init_logger
from queryLabel import Taxonomy


CATEGORIES_PATH = '/scratch/WikipediaImagesTaxonomy/commonswiki-20220220-category-network.parquet'
GRAPH_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-category-graph.pkl'
FILES_PATH = '/scratch/WikipediaImagesTaxonomy/commonswiki-20220220-files.parquet'
COMMONS_URL = 'https://commons.wikimedia.org/wiki/File:'
UPLOAD_URL = 'https://upload.wikimedia.org/wikipedia/commons/'
LOG_PATH = 'temp.log'

files_test = pd.DataFrame({'title': ['Mirror LACMA', '2020-03-03-bonn-bergheimer', 'Family Day 13 Canoe'], 
                           'url': ['f/fa/Mirror_LACMA_M.76.174.12.jpg', 
                                   '2/27/2020-03-03-bonn-bergheimer-strasse-friedhof-schwarzrheindorf-ehrenmal-1949-01.jpg', 
                                   '3/33/Family_Day_13_Canoe_9036_%289938815913%29.jpg'],
                           'categories': [['Pre-Islamic art of Iran in the Los Angeles County Museum of Art', 'Mirrors in the Los Angeles County Museum of Art'],
                                          ['2020 in Bonn', 'Gedenkkreuz Schwarzrheindorf 1949 (Bonn)'],
                                          ['Southern Arkansas University', 'Canoes in Arkansas']]})
filesize = len(files_test)

logger = init_logger(LOG_PATH, logger_name='taxonomy')
logfile = open(LOG_PATH, 'r+')

def queryFile():
    '''
    Query the current file, extracting the labels and displaying the corresponding image.
    '''
    file = files_test.iloc[st.session_state.fileList[st.session_state.counter]]
    logger.debug(f'Querying file {file.title}\n')
    labels = set()
    for category in file.categories:
        logger.debug(f'Starting search for category {category}')
        # cat_labels = st.session_state.taxonomy.get_label(category, how=st.session_state.how.lower())
        cat_labels = {f'test x {category[0:5]}'}
        logger.debug(f'Ending search for category {category} with resulting labels {cat_labels}')
        logger.debug(f'---------------------------------------------------------------------------------')
        labels |= cat_labels
    logger.debug(f'Final labels: {labels}')

    # Extract the current log
    log_dump = logfile.read().replace('\n', '  \n ')
    # Erase the log, resetting the stream position
    logfile.truncate(0)
    logger.handlers[0].stream.seek(0)

    commons_link = COMMONS_URL + file.url.split('/')[-1]
    image_link = UPLOAD_URL + file.url
    # Display the image with a link to commons
    st.markdown(f'<a href="{commons_link}"><img src="{image_link}"' +
                 ' alt="drawing" width="600"/>', unsafe_allow_html=True)

    st.subheader('Labels: [' + ', '.join(labels) + ']')
    
    with st.expander("Show log"):
        st.write(log_dump)


def nextFile():
    if(st.session_state.counter == len(st.session_state.fileList) - 1):
        st.session_state.fileList.append(np.random.randint(0, filesize))

    st.session_state.counter += 1
    st.session_state.previous_disabled = not bool(st.session_state.counter)
    queryFile()


def previousFile():
    st.session_state.counter -= 1
    if(not st.session_state.counter):
        st.session_state.previous_disabled = True
    queryFile()


def resetSeedNumber():
    np.random.seed(st.session_state.seedNumber)
    st.session_state.fileList = []
    st.session_state.previous_disabled = True
    st.session_state.counter = -1
    nextFile()


# Initializations
if 'seedNumber' not in st.session_state:
    # Initialize taxonomy
    # st.session_state.taxonomy = Taxonomy()
    # with st.spinner('Initializing taxonomy...'):
    #     st.session_state.taxonomy.load_graph(GRAPH_PATH)
    # with st.spinner('Initializing mapping...'):
    #     st.session_state.taxonomy.set_taxonomy(mapping='content_extended')
    # with st.spinner('Initializing lexical parser...'):
    #     st.session_state.taxonomy.get_head('CommonsRoot')

    # Initialize session
    st.session_state.seedNumber = 0
    # st.session_state.how = 'heuristics'
    resetSeedNumber()

    



col1, col2, col3, col4 = st.columns([1, 1, 1.8, 1.8])
col1.text(""); col1.text("");
col1.button('Previous file', key='previousb', on_click=previousFile, disabled=st.session_state.previous_disabled)
col2.text(""); col2.text("");
col2.button('Next file', on_click=nextFile)
col3.selectbox('Method', options=['All', 'Naive', 'Heuristics'], index=2, key='how', on_change=queryFile)
col4.number_input('Random seed', min_value=0, key='seedNumber', on_change=queryFile)
col4.button('Reset', on_click=resetSeedNumber)
