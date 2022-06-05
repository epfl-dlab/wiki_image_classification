import numpy as np 
import pandas as pd
import networkx as nx
import logging, sys
import pickle

from headParsing import find_head
from iteration_utilities import duplicates, unique_everseen


logger = logging.getLogger('taxonomy')


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
        G.remove_node('')
        self.G = G
    
    def dump_graph(self, path):
        '''
        Save the edge list in a file
        '''
        assert '.pkl' in path
        with open(path, 'wb') as f:
            pickle.dump(self.G, f)
    
    def load_graph(self, path):
        '''
        Load the edge list from a file
        '''
        assert '.pkl' in path
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


    def get_label(self, category, how='heuristics', debug=False):
        '''
        Get the label corresponding to a specific category, passed as string.

        Params:
            how (string): decision scheme to recursively query parents. 
                all: all parents are queried
                naive: hop only to lower-depth parents
                heuristics: decision based on the set of heuristics described in ??
        '''
        assert isinstance(category, str)

        try:
            curr_node = self.G.nodes[category]
        except KeyError:
            debug and logger.warning('Red link, skipping category ' + category)
            return set()

        # Temporary solution to non-connected categories (due to missing template expansion)
        if('depth' not in curr_node):
            debug and logger.warning(f'Non connected category {category}, returning empty set')
            return set()

        if(curr_node['visited']):
            debug and logger.debug('Found ' + category + ' with label ' + str(curr_node['labels']))
            return curr_node['labels']
        
        else:
            curr_node['visited'] = True
            self.visited_nodes += 1
            debug and logger.debug(str(self.visited_nodes) + ' - Searching for ' + category +
                          ' (depth ' + str(curr_node.get('depth', None)) + '), with parents ' +
                          str(list(self.G.neighbors(category))) + '...')

            if(how == 'all'):
                for parent in self.G.neighbors(category):
                    curr_node['labels'].update(self.get_label(parent, how))
                return curr_node['labels']

            elif(how == 'naive'):
                depth = curr_node['depth']
                for parent in self.G.neighbors(category):
                    try:
                        if(self.G.nodes[parent]['depth'] < depth):
                            curr_node['labels'].update(self.get_label(parent, how))
                    # Not connected category (temp fix to template expansion)
                    except KeyError:
                        continue
                return curr_node['labels']

            elif(how == 'heuristics' or how == 'heuristics_simple'):
                depth = curr_node['depth']

                # 1 Hidden category
                if(curr_node['hiddencat']):
                    debug and logger.debug('Hidden category, returning empty set')
                    return set()

                # 2 Lexical head

                # 2.1. Check for meaningless head (time-related + Commons-related)
                null_heads = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December',
                              'Spring', 'Summer', 'Autumn', 'Winter', 'Century', 'Categories', 'Category']
                heads = [self.get_head(category)]
                if(heads[0].isnumeric() or heads[0] in null_heads):
                    debug and logger.debug('Head ' + heads[0] + ' not meaningful, returning empty set')
                    return set()

                # Get heads of all parents
                for parent in self.G.neighbors(category):
                    heads.append(self.get_head(parent))
                debug and logger.debug('Heads: ' + str(heads))

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
                    debug and logger.debug('Lexical heads: ' + str(heads))
                debug and logger.debug('\tFound common heads: ' + str(common_heads))

                # 2.3. Hop to common_heads if they belong to parents and are not meaningless
                for common_head in common_heads:
                    if((how == 'heuristics' and common_head in nx.descendants(self.G, category) 
                        or (how == 'heuristics_simple' and taxonomy.G.nodes.get(common_head, {}).get('depth', 1e9) < depth))
                       and not (common_head.isnumeric() or common_head in null_heads)):
                        curr_node['labels'].update(self.get_label(common_head, how))
                    else:
                        debug and logger.debug('Common head ' + str(common_head) + ' not found or time-related')
                
                # Will be empty if no common_head is found, if the common_heads are
                # all not valid category names, hidden categories or already visited 
                # (including the current category)
                if(curr_node['labels']):
                    return curr_node['labels']

                # 3. is_a or subcategory_of (temp: depth check)
                for parent in self.G.neighbors(category):
                    try:
                        if(self.G.nodes[parent]['depth'] < depth):
                            curr_node['labels'].update(self.get_label(parent, how))
                        else:
                            debug and logger.debug('[' + category + '] Skipping parent ' + parent + 
                            ' (depth ' + str(self.G.nodes[parent]['depth']) + ')')
                    # Not connected category (temp fix to template expansion)
                    except KeyError:
                        debug and logger.warning('[' + category + '] Parent ' + parent + ' not connected.')
                        continue
                return curr_node['labels']

            else:
                raise ValueError('Invalid "how" option')
