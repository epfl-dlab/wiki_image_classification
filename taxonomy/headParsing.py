import stanza

def find_tree_head(tree):
    '''
    Get head of a Noun Phrase. Based on a simplified version of Michael Collins' 1999 rules.
    Returns the maximal sequence of noun-tagged words, instead of a 1-word head.
    Cf https://stackoverflow.com/questions/32654704/finding-head-of-a-noun-phrase-in-nltk-and-stanford-parse-according-to-the-rules
    '''

    # Get the first NP
    while(tree.children and tree.label != 'NP'):
        tree = tree.children[0]
    
    if(not tree.children):
        return ''
    
    # Break compound sentence
    while(tree.children[0].label == 'NP'):
        tree = tree.children[0]
    
    if(tree.children[-1].label == 'POS'):
        return tree.children[-1].leaf_labels()[0]
    
    head = []
    for child in tree.children[::-1]:

        if(child.label in ['NN', 'NNS', 'NNP', 'NNPS', 'NNS', 'POS', 'JJR']):
            head += child.leaf_labels()
        elif(child.label in ['NML']):
            head += child.leaf_labels()[::-1]
        elif(head):
            break
    
    if head:
        return ' '.join(head[::-1])

    for child in tree.children:
        if(child.label in ['ADJP', 'PRN']):
            return child.leaf_labels()[0]

    for child in tree.children:
        if(child.label in ['CD']):
            return child.leaf_labels()[0]

    for child in tree.children:
        if(child.label in ['JJ', 'JJS', 'RB', 'QP']):
            return child.leaf_labels()[0]
    
    return tree.children[-1].leaf_labels()[0]
    

def find_head(categories, use_gpu=True):
    '''
    Find the lexical head of a category, considering only takes the first sentence
    '''
    if isinstance(categories, str):
        categories = [categories]

    if not hasattr(find_head, "nlp"):
        find_head.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', use_gpu=use_gpu)

    doc = find_head.nlp('.\n\n'.join(categories))
    heads = map(lambda c: find_tree_head(c.constituency).capitalize(), doc.sentences)
    return list(heads)

