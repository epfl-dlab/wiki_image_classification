import logging
import pickle
import sys
from functools import partial
from operator import itemgetter

import networkx as nx
import numpy as np
import pandas as pd
from iteration_utilities import duplicates, unique_everseen
from scipy import spatial

sys.path.append("./")
sys.path.append("../../")

from src.config import *
from src.taxonomy.head import headParsing
from src.taxonomy.taxonomy import Taxonomy

logger = logging.getLogger("taxonomy")


class Heuristics:
    def __init__(self, G=None):
        if G:
            self.G = G

    def load_categories(self, path):
        """
        Load categories from path and build the category graph.
        """
        self.build_category_graph(pd.read_parquet(path))

    def build_category_graph(self, categories):
        """
        Build the category graph, starting from the DataFrame extracted by processing dumps
        """
        categories = categories.set_index("title")
        # Build DiGraph from adjacency matrix
        G = nx.DiGraph(categories.parents.to_dict())
        nx.set_node_attributes(
            G,
            dict(
                zip(
                    categories.index,
                    categories[["id", "hiddencat"]].to_dict(orient="records"),
                )
            ),
        )
        depth = {
            node: len(sps)
            for node, sps in nx.shortest_path(G, target="CommonsRoot").items()
        }
        nx.set_node_attributes(G, depth, name="depth")
        G.remove_node("")
        self.G = G

    def dump_graph(self, path, clean=False):
        """
        Save the edge list in a file
        """
        assert ".pkl" in path
        G = self.G if not clean else self.G_h
        with open(path, "wb") as f:
            pickle.dump(G, f)

    def load_graph(self, path, clean=False):
        """
        Load the edge list from a file
        """
        assert ".pkl" in path
        with open(path, "rb") as f:
            G = pickle.load(f)

        if clean:
            self.G_h = G
        else:
            self.G = G

    def reset_labels(self):
        """
        Reset labels and discovery status for each node.
        """
        nx.set_node_attributes(
            self.G, {node: {"visited": False, "labels": set()} for node in self.G.nodes}
        )
        self.visited_nodes = 0
        self.G_h = nx.DiGraph()

    def set_taxonomy(self, taxonomy_version):
        """
        Set an ORES-like taxonomy, mapping labels to high-level categories.
        """
        self.taxonomy_version = taxonomy_version
        self.taxonomy = Taxonomy()
        self.taxonomy.set_taxonomy(taxonomy_version)
        self.mapping = self.taxonomy.get_flat_mapping()

        self.reset_labels()
        for label, categories in self.mapping.items():
            for category in categories:
                self.visited_nodes += 1
                self.G.nodes[category]["visited"] = True
                self.G.nodes[category]["labels"].add(label)

    def set_heuristics(self, heuristics_version):
        """
        Set heuristics set.
        """
        self.heuristics_version = heuristics_version

        heuristics_list = heuristics_version.split("+")
        self.heuristics = []

        for heuristic in heuristics_list:
            if heuristic == "head":
                self.heuristics.append(
                    partial(self._head_matching, jump=False, multiple_words=False)
                )
            elif heuristic == "headJ":
                self.heuristics.append(
                    partial(self._head_matching, jump=True, multiple_words=True)
                )
            elif heuristic == "depth":
                self.heuristics.append(self._depth_check)

            elif heuristic.startswith("embedding"):
                threshold = float("0." + heuristic.split("embedding")[1])
                self.heuristics.append(
                    partial(self._embedding_similarity, threshold=threshold)
                )

            else:
                raise ValueError(f"Invalid heuristic {heuristic}")

    def get_head(self, category):
        """
        Get or compute the lexical head of a given category.
        """
        if "head" in self.G.nodes[category]:
            head = self.G.nodes[category]["head"]
        else:
            head = headParsing.find_head(category)
            self.G.nodes[category]["head"] = head
        return head

    def get_embedding(self, category):
        """
        Get the BERT sentence-embedding of a category.
        """
        if "embedding" in self.G.nodes[category]:
            return self.G.nodes[category]["embedding"]
        else:
            raise ValueError(
                f"Embedding not found for category {category}. Make sure to run the script process_embedding.py first, and to load the graph {EH_GRAPH_PATH}."
            )

    def _head_matching(self, category, jump, multiple_words, debug=False):
        """
        Head matching heuristic: parent categories are queried if
        they have the same lexical head as the current category.

        Parameters
        ----------
        category : str
            Category to be processed.
        jump : bool
            If true, the next categories to be queried are directly
            the matched heads instead of the parents with matching
            heads. The matched heads, in case of a jump, are queried
            only if their depth is lower than the current category
            (proxy for is_ancestor check, too slow to be used).
            When jumping, we allow heads composed of multiple words.
        debug : bool
            If true, print debug information.

        Returns
        ----------
        list of str
            List of next categories to query.
        """

        heads = [[self.get_head(category), category]]

        # Get heads of all parents
        parents = self.G.neighbors(category)
        for parent in parents:
            if multiple_words:
                heads.append([self.get_head(parent), parent])
            else:
                heads.append(
                    [self.get_head(parent).split(" ")[-1].capitalize(), parent]
                )
        debug and logger.debug("[HM] Heads: " + str(heads))

        # Try to match over complete lexical heads or subsets of them
        while 1:
            common_heads = list(unique_everseen(duplicates(heads, key=itemgetter(0))))

            # Break if found a common head or all the heads are already 1 word long
            if (
                common_heads
                or (cmax := max(map(lambda x: len(x[0].split()), heads))) == 1
            ):
                break

            # Remove 1 word from the longest composite heads
            for i, (head, parent) in enumerate(heads):
                head_words = head.split()
                if len(head_words) == cmax:
                    heads[i][0] = " ".join(head_words[1:]).capitalize()
            debug and logger.debug("[HM] Lexical heads: " + str(heads))
        debug and len(common_heads) > 0 and logger.debug(
            "\t[HM] Found common heads: " + str(common_heads)
        )

        # Hop to common_heads if they belong to parents
        next_queries = []
        if jump:
            for (common_head, parent) in common_heads:
                if (
                    self.G.nodes.get(common_head, {}).get("depth", 1e9)
                    < self.G.nodes[category]["depth"]
                ):
                    next_queries.append(common_head)
                else:
                    debug and logger.debug(
                        "[HM] Common head "
                        + str(common_head)
                        + " not found or too deep, skipping"
                    )
        else:
            for (head, parent) in heads:
                common_heads_heads = [common_head[0] for common_head in common_heads]
                if head in common_heads_heads:
                    next_queries.append(parent)

        return next_queries

    def _depth_check(self, category, debug=False):
        """
        Depth check heuristic: parent categories are queried if
        they are at a lower depth than the current category.

        Parameters
        ----------
        category : str
            Category to be processed.
        debug : bool
            If true, print debug information.

        Returns
        ----------
        list of str
            List of next categories to query.
        """

        depth = self.G.nodes[category]["depth"]
        next_queries = []
        for parent in self.G.neighbors(category):
            try:
                if self.G.nodes[parent]["depth"] < depth:
                    next_queries.append(parent)
                else:
                    debug and logger.debug(
                        "[DC] ["
                        + category
                        + "] Skipping parent "
                        + parent
                        + " (depth "
                        + str(self.G.nodes[parent]["depth"])
                        + ")"
                    )
            # Not connected category (temp fix to template expansion)
            except KeyError:
                debug and logger.warning(
                    "[DC] [" + category + "] Parent " + parent + " not connected."
                )
                continue
        return next_queries

    def _embedding_similarity(self, category, threshold, debug=False):
        """
        Embedding similarity heuristic: parent categories are queried if
        the cosine similarity of their embedding wrt the one of the current category
        is above a certain threshold.

        Parameters
        ----------
        category : str
            Category to be processed.
        threshold : float
            Threshold for the cosine similarity.
        debug : bool
            If true, print debug information.

        Returns
        ----------
        list of str
            List of next categories to query.
        """
        embedding = self.get_embedding(category)
        next_queries = []
        for parent in self.G.neighbors(category):
            similarity = spatial.distance.cosine(embedding, self.get_embedding(parent))
            if similarity < threshold:
                next_queries.append(parent)
                debug and logger.debug(
                    "[ES"
                    + str(threshold)[2:]
                    + "] ["
                    + category
                    + "] Found "
                    + parent
                    + " with similarity "
                    + str(similarity)
                )
            else:
                debug and logger.debug(
                    "[ES"
                    + str(threshold)[2:]
                    + "] ["
                    + category
                    + "] Skipping parent "
                    + parent
                    + " (similarity "
                    + str(similarity)
                    + ")"
                )

        return next_queries

    def query_category(self, category, debug=False):
        """
        Get the label corresponding to a specific category, passed as string.

        Params:
            how (string): decision scheme to recursively query parents.
                all: all parents are queried
                naive: hop only to lower-depth parents
                heuristics: decision based on the set of heuristics described in (Salvi, 2022)
        """
        assert isinstance(category, str)

        ## Basic checks
        # Red links
        try:
            curr_node = self.G.nodes[category]
        except KeyError:
            debug and logger.warning("Red link, skipping category " + category)
            return set()

        # Temporary solution to non-connected categories (due to missing template expansion)
        if "depth" not in curr_node:
            debug and logger.warning(
                f"Non connected category {category}, returning empty set"
            )
            return set()

        ## Base case: already visited, return labels
        if curr_node["visited"]:
            debug and logger.debug(
                "Found " + category + " with label " + str(curr_node["labels"])
            )
            return curr_node["labels"]

        curr_node["visited"] = True
        self.visited_nodes += 1
        debug and logger.debug(
            str(self.visited_nodes)
            + " - Searching for "
            + category
            + " (depth "
            + str(curr_node.get("depth", None))
            + "), with parents "
            + str(list(self.G.neighbors(category)))
            + "..."
        )

        ## More basic checks
        # Hiddencats
        if curr_node["hiddencat"]:
            debug and logger.debug("Hidden category, returning empty set")
            return set()

        # Meaningless head (time-related + Commons-related)
        null_heads = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
            # "Spring",
            # "Summer",
            # "Autumn",
            # "Winter",
            "Century",
            "Categories",
            "Category",
        ]
        head = self.get_head(category)
        if head.isnumeric() or head in null_heads:
            debug and logger.debug(
                "Head " + head + " not meaningful, returning empty set"
            )
            return set()

        ## Go through heuristics, stopping as soon as one returns a non-empty set
        for heuristic in self.heuristics:
            next_queries = heuristic(category, debug=debug)
            for next_query in next_queries:
                self.G_h.add_edge(category, next_query)
                curr_node["labels"].update(self.query_category(next_query, debug=debug))

            if curr_node["labels"]:
                break

        return curr_node["labels"]

    def queryFile(self, file, debug=False, logfile=None):
        """
        Given one file, a row of the files DataFrame, queries recursively all
        the categories and returns the final labels.
        """

        labels = set()
        for category in file.categories:
            debug and logger.debug(f"Starting search for category {category}")
            cat_labels = self.query_category(category, debug=debug)
            debug and logger.debug(
                f"Ending search for category {category} with resulting labels {cat_labels}"
            )
            debug and logger.debug(
                f"---------------------------------------------------"
            )
            labels |= cat_labels
        debug and logger.debug(f"Final labels: {labels}")
        log = logfile.read() if debug else None
        return labels, log
