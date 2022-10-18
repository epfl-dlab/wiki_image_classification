class Label:
    def __init__(self, name, categories):
        self.name = name
        self.own_categories = categories
        self.categories = categories
        self.childrens = []

    def add_children(self, children):
        queue = [children]
        while queue:
            curr = queue.pop()
            self.categories += curr.categories
            queue.extend(curr.childrens)

        self.childrens.append(children)

    def add_childrens(self, childrens):
        for children in childrens:
            self.add_children(children)


class Taxonomy:
    def __init__(self, version=None):
        if version:
            self.version = version
            self.set_taxonomy(version)

    def set_taxonomy(self, version):
        self.version = version
        if version == "v0.0":
            self.taxonomy = Label("All", [])

            nature = Label("Nature", ["Nature"])
            nature.add_childrens(
                [
                    Label("Animals", ["Animalia"]),
                    Label("Fossils", ["Fossils"]),
                    Label("Landscapes", ["Landscapes"]),
                    Label("Marine organisms", ["Marine organisms"]),
                    Label("Plants", ["Plantae"]),
                    Label("Weather", ["Weather"]),
                ]
            )
            test = Label("Test", ["Test"])
            test.add_children(Label("Test1", ["Test1"]))
            test.add_children(Label("Test2", ["Test2"]))
            nature.add_children(test)
            self.taxonomy.add_children(nature)

            society_culture = Label("Society/Culture", ["Society", "Culture"])
            society_culture.add_childrens(
                [
                    Label("Art", ["Art"]),
                    Label("Belief", ["Belief"]),
                    Label("Entertainment", ["Entertainment"]),
                    Label("Events", ["Events"]),
                    Label("Flags", ["Flags"]),
                    Label("Food", ["Food"]),
                    Label("History", ["History"]),
                    Label("Language", ["Language"]),
                    Label("Literature", ["Literature"]),
                    Label("Music", ["Music"]),
                    Label("Objects", ["Objects"]),
                    Label("People", ["People"]),
                    Label("Places", ["Places"]),
                    Label("Politics", ["Politics"]),
                    Label("Sports", ["Sports"]),
                ]
            )
            self.taxonomy.add_children(society_culture)

            science = Label("Science", ["Science"])
            science.add_childrens(
                [
                    Label("Astronomy", ["Astronomy"]),
                    Label("Biology", ["Biology"]),
                    Label("Chemistry", ["Chemistry"]),
                    Label("Earth sciences", ["Earth sciences"]),
                    Label("Mathematics", ["Mathematics"]),
                    Label("Medicine", ["Medicine"]),
                    Label("Physics", ["Physics"]),
                    Label("Technology", ["Technology"]),
                ]
            )
            self.taxonomy.add_children(science)

            engineering = Label("Engineering", ["Engineering"])
            engineering.add_childrens(
                [
                    Label("Architecture", ["Architecture"]),
                    Label("Chemical eng", ["Chemical engineering"]),
                    Label("Civil eng", ["Civil engineering"]),
                    Label("Electrical eng", ["Electrical engineering"]),
                    Label("Environmental eng", ["Environmental engineering"]),
                    Label("Geophysical eng", ["Geophysical engineering"]),
                    Label("Mechanical eng", ["Mechanical engineering"]),
                    Label("Process eng", ["Process engineering"]),
                ]
            )
            self.taxonomy.add_children(engineering)
        else:
            raise ValueError("Invalid taxonomy version")

    def get_flat_mapping(self):
        mapping = {}

        def dfs(node):
            mapping[node.name] = node.categories
            for children in node.childrens:
                dfs(children)

        dfs(self.taxonomy)
        del mapping["All"]
        return mapping

    def get_all_labels(self):
        labels = []

        def dfs(node):
            labels.append(node.name)
            for children in node.childrens:
                dfs(children)

        dfs(self.taxonomy)
        del labels[0]
        return labels

    def get_all_leafs(self):
        leafs = []

        def dfs(node):
            if not node.childrens:
                leafs.append(node.name)
            for children in node.childrens:
                dfs(children)

        dfs(self.taxonomy)
        return leafs

    def get_all_clusters(self, max_level=None):
        clusters = []

        def dfs(node, level):
            if node.childrens:
                clusters.append(node.name)
            if max_level is None or level < max_level:
                for children in node.childrens:
                    dfs(children, level + 1)

        dfs(self.taxonomy, 0)
        del clusters[0]
        return clusters
