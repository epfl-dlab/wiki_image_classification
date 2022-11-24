class Label:
    def __init__(self, name, categories, hierarchical=True):
        self.name = name
        self.own_categories = categories
        self.categories = categories
        self.children = []
        self.hierarchical = hierarchical

    def add_child(self, child):
        if self.hierarchical:
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
    def __init__(self, hierarchical=True):
        self.hierarchical = hierarchical

    def set_taxonomy(self, version):
        self.version = version
        if version == "test":
            self.taxonomy = Label("All", [], self.hierarchical)

            first = Label("First", ["first"], self.hierarchical)
            test = Label("Test", ["Test"], self.hierarchical)
            test.add_child(Label("Test1", ["Test1"], self.hierarchical))
            test.add_child(Label("Test2", ["Test2"], self.hierarchical))
            first.add_child(test)
            self.taxonomy.add_child(first)

            second = Label("Second", ["second"], self.hierarchical)
            ttest = Label("Try", ["TTest"], self.hierarchical)
            ttest.add_child(Label("TTest1", ["TTest1"]), self.hierarchical)
            ttest.add_child(Label("TTest2", ["TTest2"]), self.hierarchical)
            first.add_child(ttest)
            self.taxonomy.add_child(second)

        elif version == "v0.0":
            self.taxonomy = Label("All", [], self.hierarchical)

            nature = Label("Nature", ["Nature"], self.hierarchical)
            nature.add_children(
                [
                    Label("Animals", ["Animalia"], self.hierarchical),
                    Label("Fossils", ["Fossils"], self.hierarchical),
                    Label("Landscapes", ["Landscapes"], self.hierarchical),
                    Label("Marine organisms", ["Marine organisms"], self.hierarchical),
                    Label("Plants", ["Plantae"], self.hierarchical),
                    Label("Weather", ["Weather"], self.hierarchical),
                ]
            )
            self.taxonomy.add_child(nature)

            society_culture = Label(
                "Society/Culture", ["Society", "Culture"], self.hierarchical
            )
            society_culture.add_children(
                [
                    Label("Art", ["Art"], self.hierarchical),
                    Label("Belief", ["Belief"], self.hierarchical),
                    Label("Entertainment", ["Entertainment"], self.hierarchical),
                    Label("Events", ["Events"], self.hierarchical),
                    Label("Flags", ["Flags"], self.hierarchical),
                    Label("Food", ["Food"], self.hierarchical),
                    Label("History", ["History"], self.hierarchical),
                    Label("Language", ["Language"], self.hierarchical),
                    Label("Literature", ["Literature"], self.hierarchical),
                    Label("Music", ["Music"], self.hierarchical),
                    Label("Objects", ["Objects"], self.hierarchical),
                    Label("People", ["People"], self.hierarchical),
                    Label("Places", ["Places"], self.hierarchical),
                    Label("Politics", ["Politics"], self.hierarchical),
                    Label("Sports", ["Sports"], self.hierarchical),
                ]
            )
            self.taxonomy.add_child(society_culture)

            science = Label("Science", ["Science"], self.hierarchical)
            science.add_children(
                [
                    Label("Astronomy", ["Astronomy"], self.hierarchical),
                    Label("Biology", ["Biology"], self.hierarchical),
                    Label("Chemistry", ["Chemistry"], self.hierarchical),
                    Label("Earth sciences", ["Earth sciences"], self.hierarchical),
                    Label("Mathematics", ["Mathematics"], self.hierarchical),
                    Label("Medicine", ["Medicine"], self.hierarchical),
                    Label("Physics", ["Physics"], self.hierarchical),
                    Label("Technology", ["Technology"], self.hierarchical),
                ]
            )
            self.taxonomy.add_child(science)

            engineering = Label("Engineering", ["Engineering"], self.hierarchical)
            engineering.add_children(
                [
                    Label("Architecture", ["Architecture"], self.hierarchical),
                    Label("Chemical eng", ["Chemical engineering"], self.hierarchical),
                    Label("Civil eng", ["Civil engineering"], self.hierarchical),
                    Label(
                        "Electrical eng", ["Electrical engineering"], self.hierarchical
                    ),
                    Label(
                        "Environmental eng",
                        ["Environmental engineering"],
                        self.hierarchical,
                    ),
                    Label(
                        "Geophysical eng",
                        ["Geophysical engineering"],
                        self.hierarchical,
                    ),
                    Label(
                        "Mechanical eng", ["Mechanical engineering"], self.hierarchical
                    ),
                    Label("Process eng", ["Process engineering"], self.hierarchical),
                ]
            )
            self.taxonomy.add_child(engineering)

        elif version == "v1.1":
            self.taxonomy = Label("All", [], self.hierarchical)
            culture = Label("Culture", ["Culture"], self.hierarchical)
            culture.add_children(
                [
                    Label("History", ["History"], self.hierarchical),
                    Label("Art", ["Art"], self.hierarchical),
                    Label("Language", ["Language"], self.hierarchical),
                    Label("Music", ["Music"], self.hierarchical),
                    Label("Literature", ["Literature"], self.hierarchical),
                ]
            )
            self.taxonomy.add_child(culture)

            society = Label("Society", ["Society"], self.hierarchical)
            society.add_children(
                [
                    Label("People", ["People"], self.hierarchical),
                    Label("Sports", ["Sports"], self.hierarchical),
                    Label("Politics", ["Politics"], self.hierarchical),
                    Label("Flags", ["Flags"], self.hierarchical),
                    Label("Food", ["Food"], self.hierarchical),
                    Label("Belief", ["Belief"], self.hierarchical),
                    Label("Entertainment", ["Entertainment"], self.hierarchical),
                    # Label("Events", ["Events"]), # TODO: is "Events" even semantically useful?
                ]
            )
            self.taxonomy.add_child(society)

            stem = Label("STEM", ["STEM"], self.hierarchical)
            # First, add children that don't have any children themselves
            stem.add_children(
                [
                    Label("Architecture", ["Architecture"], self.hierarchical),
                    Label("Biology", ["Biology"], self.hierarchical),
                    Label("Physics", ["Physics"], self.hierarchical),
                    Label("Chemistry", ["Chemistry"], self.hierarchical),
                    Label("Astronomy", ["Astronomy"], self.hierarchical),
                    Label("Mathematics", ["Mathematics"], self.hierarchical),
                    Label("Earth sciences", ["Earth sciences"], self.hierarchical),
                    Label("Medicine", ["Medicine"], self.hierarchical),
                    Label("Technology", ["Technology"], self.hierarchical),
                    # Label("Engineering", ["Engineering"]), # TODO: remove this and keep "Technology"?
                ]
            )
            # Now, create Nature, which is a child of STEM, add its children, and add it to STEM
            nature = Label("Nature", ["Nature"], self.hierarchical)
            nature.add_children(
                [
                    Label("Animals", ["Animalia"], self.hierarchical),
                    Label("Fossils", ["Fossils"], self.hierarchical),
                    Label("Plants", ["Plantae"], self.hierarchical),
                    Label("Weather", ["Weather"], self.hierarchical),
                    Label("Landscapes", ["Landscapes"], self.hierarchical),
                    # Label("Marine organisms", ["Marine organisms"]), # TODO: is this useful?
                ]
            )
            stem.add_child(nature)
            self.taxonomy.add_child(stem)

        elif version == "v1.2":
            self.taxonomy = Label("All", [], self.hierarchical)

            stem = Label("STEM", ["STEM"], self.hierarchical)
            stem.add_children(
                [
                    Label("Biology", ["Biology"], self.hierarchical),
                    Label("Physics", ["Physics"], self.hierarchical),
                    Label("Chemistry", ["Chemistry"], self.hierarchical),
                    Label("Astronomy", ["Astronomy"], self.hierarchical),
                    Label("Mathematics", ["Mathematics"], self.hierarchical),
                    Label("Earth sciences", ["Earth sciences"], self.hierarchical),
                    Label("Medicine", ["Medicine"], self.hierarchical),
                    Label("Technology", ["Technology"], self.hierarchical),
                    Label("Engineering", ["Engineering"], self.hierarchical),
                ]
            )

            nature = Label("Nature", ["Nature"], self.hierarchical)
            nature.add_children(
                [
                    Label("Animals", ["Animalia"], self.hierarchical),
                    Label("Fossils", ["Fossils"], self.hierarchical),
                    Label("Plants", ["Plantae"], self.hierarchical),
                    Label(
                        "Weather and climate",
                        ["Weather and climate"],
                        self.hierarchical,
                    ),
                ]
            )
            stem.add_child(nature)
            self.taxonomy.add_child(stem)

            places = Label("Places", ["Places"], self.hierarchical)
            places.add_children(
                [
                    Label("Architecture", ["Architecture"], self.hierarchical),
                    Label("Landscapes", ["Landscapes"], self.hierarchical),
                    Label("Maps", ["Maps"], self.hierarchical),
                ]
            )
            self.taxonomy.add_child(places)

            society = Label("Society", ["Society"], self.hierarchical)
            society.add_children(
                [
                    Label("People", ["People"], self.hierarchical),
                    Label("Sports", ["Sports"], self.hierarchical),
                    Label("Politics", ["Politics"], self.hierarchical),
                    Label("Events", ["Events"], self.hierarchical),
                    Label("Entertainment", ["Entertainment"], self.hierarchical),
                    Label("Flags", ["Flags"], self.hierarchical),
                ]
            )
            self.taxonomy.add_child(society)

            culture = Label("Culture", ["Culture"], self.hierarchical)
            culture.add_children(
                [
                    Label("History", ["History"], self.hierarchical),
                    Label("Art", ["Art"], self.hierarchical),
                    Label("Language", ["Language"], self.hierarchical),
                    Label("Music", ["Music"], self.hierarchical),
                    Label("Literature", ["Literature"], self.hierarchical),
                    Label("Food", ["Food"], self.hierarchical),
                    Label("Belief", ["Belief"], self.hierarchical),
                ]
            )
            self.taxonomy.add_child(culture)

        elif version == "v1.3":
            self.taxonomy = Label("All", [], self.hierarchical)

            stem = Label("STEM", ["STEM"], self.hierarchical)

            natural_sciences = Label(
                "Natural sciences", ["Natural sciences"], self.hierarchical
            )
            natural_sciences.add_children(
                [
                    Label("Mathematics", ["Mathematics"], self.hierarchical),
                    Label("Chemistry", ["Chemistry"], self.hierarchical),
                    Label("Astronomy", ["Astronomy"], self.hierarchical),
                ]
            )
            stem.add_child(natural_sciences)

            stem.add_children(
                [
                    Label("Medicine", ["Medicine"], self.hierarchical),
                    Label("Technology", ["Technology"], self.hierarchical),
                ]
            )

            nature = Label("Nature", ["Nature"], self.hierarchical)
            nature.add_children(
                [
                    Label("Plants", ["Plantae"], self.hierarchical),
                    Label("Animals", ["Animalia"], self.hierarchical),
                    Label("Fossils", ["Fossils"], self.hierarchical),
                    Label(
                        "Weather and climate",
                        ["Weather and climate"],
                        self.hierarchical,
                    ),
                ]
            )
            stem.add_child(nature)
            self.taxonomy.add_child(stem)

            places = Label("Places", ["Places"], self.hierarchical)
            places.add_children(
                [
                    Label("Architecture", ["Architecture"], self.hierarchical),
                    Label("Landscapes", ["Landscapes"], self.hierarchical),
                    Label("Maps", ["Maps"], self.hierarchical),
                ]
            )
            self.taxonomy.add_child(places)

            society = Label("Society", ["Society"], self.hierarchical)
            society.add_children(
                [
                    Label("People", ["People"], self.hierarchical),
                    Label("Sports", ["Sports"], self.hierarchical),
                    Label("Politics", ["Politics"], self.hierarchical),
                    Label("Events", ["Events"], self.hierarchical),
                    Label("Games", ["Games"], self.hierarchical),
                    Label("Flags", ["Flags"], self.hierarchical),
                    Label("Transportation", ["Transport"], self.hierarchical),
                ]
            )
            self.taxonomy.add_child(society)

            culture = Label("Culture", ["Culture"], self.hierarchical)
            culture.add_children(
                [
                    Label("History", ["History"], self.hierarchical),
                    Label("Art", ["Art"], self.hierarchical),
                    Label("Music", ["Music"], self.hierarchical),
                    Label("Literature", ["Literature"], self.hierarchical),
                    Label("Food", ["Food"], self.hierarchical),
                    Label("Belief", ["Belief"], self.hierarchical),
                ]
            )
            self.taxonomy.add_child(culture)

        else:
            raise ValueError("Invalid taxonomy version")

        return self.taxonomy

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
