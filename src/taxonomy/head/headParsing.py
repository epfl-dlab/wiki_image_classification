import stanza


def find_tree_head(tree, multiple_words=False):
    """
    Get head of a Noun Phrase. Based on a simplified version of Michael Collins' 1999 rules.
    Returns the maximal sequence of noun-tagged words, instead of a 1-word head.
    Cf https://stackoverflow.com/questions/32654704/finding-head-of-a-noun-phrase-in-nltk-and-stanford-parse-according-to-the-rules
    """

    # Get the first NP
    while tree.children and tree.label != "NP":
        tree = tree.children[0]

    if not tree.children:
        return ""

    # Break compound sentence
    while tree.children[0].label == "NP":
        tree = tree.children[0]

    if tree.children[-1].label == "POS":
        return tree.children[-1].leaf_labels()[0]

    head = []
    for child in tree.children[::-1]:
        if child.label in ["NN", "NNS", "NNP", "NNPS", "NNS", "POS", "JJR"]:
            if multiple_words:
                head += child.leaf_labels()
            else:
                return child.leaf_labels()[0]
        elif child.label in ["NML"]:
            if multiple_words:
                head += child.leaf_labels()[::-1]
            else:
                return child.leaf_labels()[0]
        elif head:
            break

    if head:
        return " ".join(head[::-1])

    for child in tree.children:
        if child.label in ["ADJP", "PRN"]:
            return child.leaf_labels()[0]

    for child in tree.children:
        if child.label in ["CD"]:
            return child.leaf_labels()[0]

    for child in tree.children:
        if child.label in ["JJ", "JJS", "RB", "QP"]:
            return child.leaf_labels()[0]

    return tree.children[-1].leaf_labels()[0]


def align_sentences(categories, sentences):
    """
    Align categories and sentences.
    Since stanza 1.4. does not support batch processing, the official advice is to concatenate documents
    with "\n\n". However, this does not respect the boundaries of the original documents, producing potentially more sentences.
    The function matches categories with sentences, by picking the first sentence contained in the category.
    """
    new_sentences = []
    i_sent = 0
    for i in range(len(categories) - 1):
        # Needed for cases where category is split in 3+ sentences and only the 2nd is also contained in the following category
        while not sentences[i_sent].text.lower() in categories[i].lower():
            i_sent += 1
        new_sentences.append(sentences[i_sent])
        i_sent += 1

        while (
            sentences[i_sent].text.lower() in categories[i].lower()
            and sentences[i_sent].text.lower() not in categories[i + 1].lower()
        ):
            i_sent += 1

    # Last category
    while not sentences[i_sent].text.lower() in categories[-1].lower():
        i_sent += 1
    new_sentences.append(sentences[i_sent])
    assert sentences[i_sent].text.lower() in categories[-1].lower()
    assert len(categories) == len(new_sentences)
    return new_sentences


def find_head(categories, multiple_words=False, use_gpu=True):
    """
    Find the lexical head of a category, considering only takes the first sentence
    """
    if isinstance(categories, str):
        categories = [categories]

    if not hasattr(find_head, "nlp"):
        find_head.nlp = stanza.Pipeline(
            lang="en", processors="tokenize,pos,constituency", use_gpu=use_gpu
        )

    doc = find_head.nlp(f"\n\n".join(categories))
    sentences = align_sentences(categories, doc.sentences)
    heads = list(
        map(
            lambda c: find_tree_head(c.constituency, multiple_words).capitalize(),
            sentences,
        )
    )
    assert len(heads) == len(categories)
    return heads
