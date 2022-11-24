### Parsing paths

# list of chunks of the WIT dataset
WIT_DATASET = [
    f"/scratch/WIT_Dataset/wit_v1.train.all-0000{str(i)}-of-00010.tsv.gz"
    for i in range(0, 10)
]

# Dumps (input to generateCategories)
COMMONS_DUMP = "/scratch/WikipediaImagesTaxonomy/dumps/commonswiki-20220220-pages-articles-multistream.xml.bz2"
CATEGORIES_DUMP = "/scratch/WikipediaImagesTaxonomy/dumps/commonswiki-20220220-categories-multistream.xml.bz2"

# Parsing utilities
FILE_REDIRECTS = "/scratch/WikipediaImagesTaxonomy/file_redirects.pkl"
WIT_NAMES_PATH = "/scratch/WikipediaImagesTaxonomy/wit_names.parquet"

# Parsing outputs
CATEGORIES_PATH = (
    "/scratch/WikipediaImagesTaxonomy/commonswiki-20220220-category-network.parquet"
)
FILES_PATH = "/scratch/WikipediaImagesTaxonomy/commonswiki-20220220-files.parquet"
####################################

### Graph paths
# Graph
GRAPH_PATH = "/scratch/WikipediaImagesTaxonomy/20220220-category-graph.pkl.bz2"
# Graph enriched with heads
HGRAPH_PATH = "/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wheads.pkl.bz2"
# Graph enriched with heads and embeddings
EH_GRAPH_PATH = "/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wheads-wembeddings.pkl.bz2"
####################################

### Chunks paths
# Heads chunks
HEADS_PATH = "/scratch/WikipediaImagesTaxonomy/heads/"
# Embeddings chunks
EMBEDDINGS_PATH = "/scratch/WikipediaImagesTaxonomy/embeddings/"
####################################

# Streamlit
STREAMLIT_PATH = "./data/streamlit/"
STREAMLIT_LOG_FILE = "streamlit_preparation.log"


### Versions (!!!)
TAXONOMY_VERSION = "v1.3"
HEURISTICS_VERSION = "headJ+depth"
####################################

# Graph with labels (i.e. enriched with heads, embeddings and labels)
LGRAPH_PATH = f"/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wlabels_{HEURISTICS_VERSION}_{TAXONOMY_VERSION}.pkl.bz2"
LGRAPH_H_PATH = f"/scratch/WikipediaImagesTaxonomy/20220220-clean-graph-wlabels_{HEURISTICS_VERSION}_{TAXONOMY_VERSION}.pkl.bz2"
