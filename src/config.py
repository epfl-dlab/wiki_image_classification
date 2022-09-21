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


# Hierarchy of labels
HIERARCHICAL_TAXONOMY = {
    "Nature": [
        "Nature",
        "Animals",
        "Fossils",
        "Landscapes",
        "Marine organisms",
        "Plants",
        "Weather",
    ],
    "Society/Culture": [
        "Society",
        "Culture",
        "Art",
        "Belief",
        "Entertainment",
        "Events",
        "Flags",
        "Food",
        "History",
        "Language",
        "Literature",
        "Music",
        "Objects",
        "People",
        "Places",
        "Politics",
        "Sports",
    ],
    "Science": [
        "Science",
        "Astronomy",
        "Biology",
        "Chemistry",
        "Earth sciences",
        "Mathematics",
        "Medicine",
        "Physics",
        "Technology",
    ],
    "Engineering": [
        "Engineering",
        "Architecture",
        "Chemical eng",
        "Civil eng",
        "Electrical eng",
        "Environmental eng",
        "Geophysical eng",
        "Mechanical eng",
        "Process eng",
    ],
}
ALL_LABELS = [label for type in HIERARCHICAL_TAXONOMY.values() for label in type]

# Mapping of labels to categories
FULL_MAPPING = {
    # Nature
    "Nature": ["Nature"],
    "Animals": ["Animalia"],
    "Fossils": ["Fossils"],
    "Landscapes": ["Landscapes"],
    "Marine organisms": ["Marine organisms"],
    "Plants": ["Plantae"],
    "Weather": ["Weather"],
    # Society/Culture
    "Society": ["Society"],
    "Culture": ["Culture"],
    "Art": ["Art"],
    "Belief": ["Belief"],
    "Entertainment": ["Entertainment"],
    "Events": ["Events"],
    "Flags": ["Flags"],
    "Food": ["Food"],
    "History": ["History"],
    "Language": ["Language"],
    "Literature": ["Literature"],
    "Music": ["Music"],
    "Objects": ["Objects"],
    "People": ["People"],
    "Places": ["Places"],
    "Politics": ["Politics"],
    "Sports": ["Sports"],
    # Science
    "Science": ["Science"],
    "Astronomy": ["Astronomy"],
    "Biology": ["Biology"],
    "Chemistry": ["Chemistry"],
    "Earth sciences": ["Earth sciences"],
    "Mathematics": ["Mathematics"],
    "Medicine": ["Medicine"],
    "Physics": ["Physics"],
    "Technology": ["Technology"],
    # Engineering
    "Engineering": ["Engineering"],
    "Architecture": ["Architecture"],
    "Chemical eng": ["Chemical engineering"],
    "Civil eng": ["Civil engineering"],
    "Electrical eng": ["Electrical engineering"],
    "Environmental eng": ["Environmental engineering"],
    "Geophysical eng": ["Geophysical engineering"],
    "Mechanical eng": ["Mechanical engineering"],
    "Process eng": ["Process engineering"],
}
