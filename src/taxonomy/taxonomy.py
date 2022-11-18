class Label:
    def __init__(self, name, categories):
        self.name = name
        self.own_categories = categories
        self.categories = categories
        self.children = []

    def add_child(self, child):
        queue = [child]
        while queue:
            curr = queue.pop()
            self.categories += curr.categories
            queue.extend(curr.children)

        self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.add_child(child)


class Taxonomy:
    def __init__(self, version=None):
        if version:
            self.version = version
            self.set_taxonomy(version)

    def set_taxonomy(self, version):
        self.version = version
        if version == "test":
            self.taxonomy = Label("All", [])

            first = Label("First", ["first"])
            test = Label("Test", ["Test"])
            test.add_child(Label("Test1", ["Test1"]))
            test.add_child(Label("Test2", ["Test2"]))
            first.add_child(test)
            self.taxonomy.add_child(first)

            second = Label("Second", ["second"])
            ttest = Label("Try", ["TTest"])
            ttest.add_child(Label("TTest1", ["TTest1"]))
            ttest.add_child(Label("TTest2", ["TTest2"]))
            first.add_child(ttest)
            self.taxonomy.add_child(second)

        elif version == "v0.0":
            self.taxonomy = Label("All", [])

            nature = Label("Nature", ["Nature"])
            nature.add_children(
                [
                    Label("Animals", ["Animalia"]),
                    Label("Fossils", ["Fossils"]),
                    Label("Landscapes", ["Landscapes"]),
                    Label("Marine organisms", ["Marine organisms"]),
                    Label("Plants", ["Plantae"]),
                    Label("Weather", ["Weather"]),
                ]
            )
            self.taxonomy.add_child(nature)

            society_culture = Label("Society/Culture", ["Society", "Culture"])
            society_culture.add_children(
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
            self.taxonomy.add_child(society_culture)

            science = Label("Science", ["Science"])
            science.add_children(
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
            self.taxonomy.add_child(science)

            engineering = Label("Engineering", ["Engineering"])
            engineering.add_children(
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
            self.taxonomy.add_child(engineering)

        elif version == "v1.1":
            self.taxonomy = Label("All", [])
            culture = Label("Culture", ["Culture"])
            culture.add_children(
                [
                    Label("History", ["History"]),
                    Label("Art", ["Art"]),
                    Label("Language", ["Language"]),
                    Label("Music", ["Music"]),
                    Label("Literature", ["Literature"]),
                ]
            )
            self.taxonomy.add_child(culture)

            society = Label("Society", ["Society"])
            society.add_children(
                [
                    Label("People", ["People"]),
                    Label("Sports", ["Sports"]),
                    Label("Politics", ["Politics"]),
                    Label("Flags", ["Flags"]),
                    Label("Food", ["Food"]),
                    Label("Belief", ["Belief"]),
                    Label("Entertainment", ["Entertainment"]),
                    # Label("Events", ["Events"]), # TODO: is "Events" even semantically useful?
                ]
            )
            self.taxonomy.add_child(society)

            stem = Label("STEM", ["STEM"])
            # First, add children that don't have any children themselves
            stem.add_children(
                [
                    Label("Architecture", ["Architecture"]),
                    Label("Biology", ["Biology"]),
                    Label("Physics", ["Physics"]),
                    Label("Chemistry", ["Chemistry"]),
                    Label("Astronomy", ["Astronomy"]),
                    Label("Mathematics", ["Architecture"]),
                    Label("Earth sciences", ["Earth sciences"]),
                    Label("Medicine", ["Architecture"]),
                    Label("Technology", ["Technology"]),
                    # Label("Engineering", ["Engineering"]), # TODO: remove this and keep "Technology"?
                ]
            )
            # Now, create Nature, which is a child of STEM, add its children, and add it to STEM
            nature = Label("Nature", ["Nature"])
            nature.add_children(
                [
                    Label("Animals", ["Animalia"]),
                    Label("Fossils", ["Fossils"]),
                    Label("Plants", ["Plantae"]),
                    Label("Weather", ["Weather"]),
                    Label("Landscapes", ["Landscapes"]),
                    # Label("Marine organisms", ["Marine organisms"]), # TODO: is this useful?
                ]
            )
            stem.add_child(nature)
            self.taxonomy.add_child(stem)

        elif version == "v1.2":
            self.taxonomy = Label("All", [])

            stem = Label("STEM", ["STEM"])
            stem.add_children(
                [
                    Label("Biology", ["Biology"]),
                    Label("Physics", ["Physics"]),
                    Label("Chemistry", ["Chemistry"]),
                    Label("Astronomy", ["Astronomy"]),
                    Label("Mathematics", ["Architecture"]),
                    Label("Earth sciences", ["Earth sciences"]),
                    Label("Medicine", ["Architecture"]),
                    Label("Technology", ["Technology"]),
                    Label(
                        "Engineering", ["Engineering"]
                    ),  # TODO: remove this and keep "Technology"?
                ]
            )

            nature = Label("Nature", ["Nature"])
            nature.add_children(
                [
                    Label("Animals", ["Animalia"]),
                    Label("Fossils", ["Fossils"]),
                    Label("Plants", ["Plantae"]),
                    Label("Weather and climate", ["Weather and climate"]),
                    # Label("Marine organisms", ["Marine organisms"]), # TODO: is this useful?
                ]
            )
            stem.add_child(nature)
            self.taxonomy.add_child(stem)

            places = Label("Places", ["Places"])
            places.add_children(
                [
                    Label("Architecture", ["Architecture"]),
                    Label("Landscapes", ["Landscapes"]),
                    Label("Maps", ["Maps"]),
                ]
            )
            self.taxonomy.add_child(places)

            society = Label("Society", ["Society"])
            society.add_children(
                [
                    Label("People", ["People"]),
                    Label("Sports", ["Sports"]),
                    Label("Politics", ["Politics"]),
                    Label("Events", ["Events"]),
                    Label("Entertainment", ["Entertainment"]),
                    Label("Flags", ["Flags"]),
                ]
            )
            self.taxonomy.add_child(society)

            culture = Label("Culture", ["Culture"])
            culture.add_children(
                [
                    Label("History", ["History"]),
                    Label("Art", ["Art"]),
                    Label("Language", ["Language"]),
                    Label("Music", ["Music"]),
                    Label("Literature", ["Literature"]),
                    Label("Food", ["Food"]),
                    Label("Belief", ["Belief"]),
                ]
            )
            self.taxonomy.add_child(culture)

        else:
            raise ValueError("Invalid taxonomy version")

    def get_flat_mapping(self):
        mapping = {}

        def dfs(node):
            mapping[node.name] = node.categories
            for children in node.children:
                dfs(children)

        dfs(self.taxonomy)
        del mapping["All"]
        return mapping

    def get_all_labels(self):
        labels = []

        def dfs(node):
            labels.append(node.name)
            for children in node.children:
                dfs(children)

        dfs(self.taxonomy)
        del labels[0]
        return labels

    def get_all_leafs_labels(self):
        leafs = []

        def dfs(node):
            if not node.children:
                leafs.append(node.name)
            for children in node.children:
                dfs(children)

        dfs(self.taxonomy)
        return leafs

    def get_all_clusters(self, max_level=None):
        clusters = []

        def dfs(node, level):
            if node.children:
                clusters.append(node.name)
            if max_level is None or level < max_level:
                for children in node.children:
                    dfs(children, level + 1)

        dfs(self.taxonomy, 0)
        del clusters[0]
        return clusters
