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

# Graph
GRAPH_PATH = "/scratch/WikipediaImagesTaxonomy/20220220-category-graph.pkl.bz2"
# Graph enriched with heads
HGRAPH_PATH = "/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wheads.pkl.bz2"
# Graph labeled
# LGRAPH_PATH = '/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wlabels_heuristics_v1.0.pkl.bz2'
LGRAPH_PATH = "/scratch/WikipediaImagesTaxonomy/20220220-category-graph-wlabels_heuristics_simple_v1.0.pkl.bz2"
LGRAPH_H_PATH = "/scratch/WikipediaImagesTaxonomy/20220220-clean-graph-wlabels_heuristics_simple_v1.0.pkl.bz2"

HEADS_PATH = "/scratch/WikipediaImagesTaxonomy/heads/"

STREAMLIT_PATH = "./data/streamlit/"
STREAMLIT_LOG_FILE = "streamlit_preparation.log"


TAXONOMY_VERSION = "v0.0"
