import numpy as np 
import pandas as pd
import networkx as nx
import logging, sys
import pickle

from headParsing import find_head
from iteration_utilities import duplicates, unique_everseen


logger = logging.getLogger()


class Taxonomy:
    def __init__(self, G=None):
        if(G):
            self.G = G

    def load_categories(self, path):
        '''
        Load categories from path and build the category graph.
        '''
        self.build_category_graph(pd.read_parquet(path))
    
    def build_category_graph(self, categories):
        '''
        Build the category graph, starting from the DataFrame extracted by processing dumps
        '''
        categories = categories.set_index('title')
        # Build DiGraph from adjacency matrix
        G = nx.DiGraph(categories.parents.to_dict())
        nx.set_node_attributes(G, dict(zip(categories.index, categories[['id', 'hiddencat']].to_dict(orient='records'))))
        depth = {node: len(sps) for node, sps in nx.shortest_path(G, target='CommonsRoot').items()}
        nx.set_node_attributes(G, depth, name='depth')
        self.G = G
    
    def dump_graph(self, path):
        '''
        Save the edge list in a file
        '''
        assert path.endswith('.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.G, f)
    
    def load_graph(self, path):
        '''
        Load the edge list from a file
        '''
        assert path.endswith('.pkl')
        with open(path, 'rb') as f:
            self.G = pickle.load(f)

    def reset_labels(self):
        '''
        Reset labels and discovery status for each node.
        '''
        nx.set_node_attributes(self.G, {node: {'visited': False, 'labels': set()} for node in self.G.nodes})
        self.visited_nodes = 0

    def set_taxonomy(self, mapping='content_extended'):
        '''
        Set an ORES-like taxonomy, mapping labels to high-level categories.
        '''
        assert isinstance(mapping, dict) or isinstance(mapping, str)

        if(isinstance(mapping, dict)):
            self.mapping = mapping

        elif(mapping == 'content_general'):
            self.mapping = {'Nature': ['Animalia', 'Fossils', 'Landscapes', 'Marine organisms', 'Plantae', 'Weather'],
                            'Society/Culture': ['Art', 'Belief', 'Entertainment', 'Events', 'Flags', 'Food', 'History', 
                                                'Language', 'Literature', 'Music', 'Objects', 'People', 'Places', 'Politics', 'Sports'],
                            'Science': ['Astronomy', 'Biology', 'Chemistry', 'Earth sciences', 'Mathematics',
                                        'Medicine', 'Physics', 'Technology'],
                            'Engineering': ['Architecture', 'Chemical engineering', 'Civil engineering', 'Electrical engineering',
                                            'Environmental engineering', 'Geophysical engineering', 'Mechanical engineering', 'Process engineering']}

        elif(mapping == 'content_extended'):
            self.mapping = {# Nature
                            'Nature': ['Nature'],
                            'Animals': ['Animalia'],
                            'Fossils': ['Fossils'],
                            'Landscapes': ['Landscapes'],
                            'Marine organisms': ['Marine organisms'],
                            'Plants': ['Plantae'],
                            'Weather': ['Weather'],
                            # Society/Culture
                            'Society': ['Society'],
                            'Culture': ['Culture'],
                            'Art': ['Art'],
                            'Belief': ['Belief'],
                            'Entertainment': ['Entertainment'],
                            'Events': ['Events'],
                            'Flags': ['Flags'],
                            'Food': ['Food'],
                            'History': ['History'],
                            'Language': ['Language'],
                            'Literature': ['Literature'],
                            'Music': ['Music'],
                            'Objects': ['Objects'],
                            'People': ['People'],
                            'Places': ['Places'],
                            'Politics': ['Politics'],
                            'Sports': ['Sports'],
                            # Science
                            'Science': ['Science'],
                            'Astronomy': ['Astronomy'],
                            'Biology': ['Biology'],
                            'Chemistry': ['Chemistry'],
                            'Earth sciences': ['Earth sciences'],
                            'Mathematics': ['Mathematics'],
                            'Medicine': ['Medicine'],
                            'Physics': ['Physics'],
                            'Technology': ['Technology'],
                            # Engineering
                            'Engineering': ['Engineering'],
                            'Architecture': ['Architecture'],
                            'Chemical eng': ['Chemical engineering'],
                            'Civil eng': ['Civil engineering'],
                            'Electrical eng': ['Electrical engineering'],
                            'Environmental eng': ['Environmental engineering'],
                            'Geophysical eng': ['Geophysical engineering'],
                            'Mechanical eng': ['Mechanical engineering'],
                            'Process eng': ['Process engineering']
                            }
        else:
            raise ValueError('Invalid mapping')

        self.reset_labels()
        for label, categories in self.mapping.items():
            for category in categories:
                self.visited_nodes += 1
                self.G.nodes[category]['visited'] = True
                self.G.nodes[category]['labels'].add(label)
    
    def get_head(self, category):
        '''
        Get or compute the lexical head of a given category.
        '''
        if('head' in self.G.nodes[category]):
            head = self.G.nodes[category]['head']
        else:
            head = find_head(category)
            self.G.nodes[category]['head'] = head
        return head


    def get_label(self, category, how='heuristics'):
        '''
        Get the label corresponding to a specific category, passed as string.

        Params:
            how (string): decision scheme to recursively query parents. 
                all: all parents are queried
                naive: hop only to lower-depth parents
                heuristics: decision based on the set of heuristics described in ??
        '''
        assert isinstance(category, str)

        if(self.G.nodes[category]['visited']):
            logging.debug('Found ' + category + ' with label ' + str(self.G.nodes[category]['labels']))
            return self.G.nodes[category]['labels']
        
        else:
            self.G.nodes[category]['visited'] = True
            self.visited_nodes += 1
            logging.debug(str(self.visited_nodes) + ' - Searching for ' + category +
                          ' (depth ' + str(self.G.nodes[category]['depth']) + '), with parents ' +
                          str(list(self.G.neighbors(category))) + '...')

            if(how == 'all'):
                for parent in self.G.neighbors(category):
                    self.G.nodes[category]['labels'].update(self.get_label(parent, how))
                return self.G.nodes[category]['labels']

            elif(how=='naive'):
                # non-connected categories
                if('depth' not in self.G.nodes[category]):
                    return set()
    
                depth = self.G.nodes[category]['depth']
                for parent in self.G.neighbors(category):
                    try:
                        if(self.G.nodes[parent]['depth'] < depth):
                            self.G.nodes[category]['labels'].update(self.get_label(parent, how))
                    # Not connected category (temp fix to template expansion)
                    except KeyError:
                        continue
                return self.G.nodes[category]['labels']

            elif(how=='heuristics'):

                # 0 Temporary solution to non-connected categories (due to missing template expansion)
                if('depth' not in self.G.nodes[category]):
                    logging.exception('Non connected category, returning empty set')
                    return set()

                # 1 Hidden category
                if(self.G.nodes[category]['hiddencat']):
                    logging.debug('Hidden category, returning empty set')
                    return set()

                # 2 Lexical head

                # 2.1. Check for meaningless head (time-related + Commons-related)
                null_heads = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December',
                              'Spring', 'Summer', 'Autumn', 'Winter', 'Century', 'Categories', 'Category']
                heads = [self.get_head(category)]
                if(heads[0].isnumeric() or heads[0] in null_heads):
                    logging.debug('Head ' + heads[0] + ' not meaningful, returning empty set')
                    return set()

                # Get heads of all parents
                for parent in self.G.neighbors(category):
                    heads.append(self.get_head(parent))
                logging.debug('Heads: ' + str(heads))

                # 2.2. Try to match over complete lexical heads or subsets
                while(1):
                    common_heads = list(unique_everseen(duplicates(heads)))

                    # Break if found a common head or all the heads are already 1 word long
                    if(common_heads or (cmax:=max(map(lambda x: len(x.split()), heads))) == 1):
                        break

                    # Remove 1 word from the longest composite heads
                    for i, head in enumerate(heads):
                        head_words = head.split()
                        if(len(head_words) == cmax):
                            heads[i] = ' '.join(head_words[1:]).capitalize()
                    logging.debug('Lexical heads: ' + str(heads))
                logging.debug('\tFound common heads: ' + str(common_heads))

                # 2.3. Hop to common_heads if they belong to parents and are not meaningless
                for common_head in common_heads:
                    if(common_head in nx.descendants(self.G, category) and 
                       not (common_head.isnumeric() or common_head in null_heads)):
                        self.G.nodes[category]['labels'].update(self.get_label(common_head, how))
                    else:
                        logging.debug('Common head ' + str(common_head) + ' not found or time-related')
                
                # Will be empty if no common_head is found, if the common_heads are
                # all not valid category names, hidden categories or already visited 
                # (including the current category)
                if(self.G.nodes[category]['labels']):
                    return self.G.nodes[category]['labels']

                # 3. is_a or subcategory_of (temp: depth check)
                depth = self.G.nodes[category]['depth']
                for parent in self.G.neighbors(category):
                    try:
                        if(self.G.nodes[parent]['depth'] < depth):
                            self.G.nodes[category]['labels'].update(self.get_label(parent, how))
                        else:
                            logging.debug('[' + category + '] Skipping parent ' + parent + 
                            ' (depth ' + str(self.G.nodes[parent]['depth']) + ')')
                    # Not connected category (temp fix to template expansion)
                    except KeyError:
                        logging.exception('[' + category + '] Parent ' + parent + ' not connected.')
                        continue
                return self.G.nodes[category]['labels']

            else:
                raise ValueError('Invalid "how" option')
