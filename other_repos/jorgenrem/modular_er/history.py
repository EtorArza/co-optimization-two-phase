

"""
History decorator for DEAP with addition of `summary_function` support
"""
from collections import deque


def summary_function(ind):
    """Summarize individual into dictionary for smaller storage needs"""
    result = {}
    result['fitness'] = ind.fitness.values[0]
    result['non_movable'] = len([m for m in ind.spawned_morph if not m.joint])
    result['movable'] = len([m for m in ind.spawned_morph if m.joint])
    return result


class History(object):
    def __init__(self, func=None):
        self.genealogy_index = 0
        self.genealogy_history = dict()
        self.genealogy_tree = dict()
        self.generation_history = dict()
        self._func = summary_function if func is None else func

    def update(self, individuals):
        """Update the history with the new *individuals*. The index present in
        their :attr:`history_index` attribute will be used to locate their
        parents, it is then modified to a unique one to keep track of those
        new individuals. This method should be called on the individuals after
        each variation.
        :param individuals: The list of modified individuals that shall be
                            inserted in the history.
        If the *individuals* do not have a :attr:`history_index` attribute,
        the attribute is added and this individual is considered as having no
        parent. This method should be called with the initial population to
        initialize the history.
        Modifying the internal :attr:`genealogy_index` of the history or the
        :attr:`history_index` of an individual may lead to unpredictable
        results and corruption of the history.
        """
        try:
            parent_indices = set(ind.history_index for ind in individuals)
        except AttributeError:
            parent_indices = set()

        for ind in individuals:
            self.genealogy_index += 1
            ind.history_index = self.genealogy_index
            self.genealogy_tree[self.genealogy_index] = parent_indices

    def register(self, generation, individuals):
        """Register additional information about individuals after fitness
        evaluation"""
        for ind in individuals:
            if ind.history_index not in self.genealogy_history:
                self.genealogy_history[ind.history_index] = self._func(ind)
                self.generation_history[ind.history_index] = generation

    def genealogy(self, individual, prune=False):
        """Generate the genealogy of the given individual"""
        tree = {}
        visited = set()
        queue = deque([individual.history_index])
        while queue:
            index = queue.popleft()
            if index in self.genealogy_tree:
                parents = self.genealogy_tree[index]
                tree[index] = set(parents)
                for parent in parents:
                    if parent not in visited:
                        visited.add(parent)
                        queue.append(parent)
        if prune:
            tree = self.prune(tree)
        return tree

    def network(self, genealogy=None, prune=True):
        """Create networkx from the given genealogy or whole history"""
        import networkx
        # If no genealogy history exist there is no reason to create a network
        # graph
        if not self.genealogy_history:
            return None
        tree = genealogy if genealogy else self.genealogy_tree
        if prune:
            tree = self.prune(tree)
        graph = networkx.DiGraph(tree)
        # Iterate graph to add metadata
        for node in graph.nodes:
            metadata = self.genealogy_history[node]
            for key, val in metadata.items():
                graph.nodes[node][key] = val
            graph.nodes[node]['generation'] = self.generation_history[node]
        return graph

    def prune(self, genealogy):
        """Prune the genealogy based on information in `genealogy_history`"""
        # If no history exist we return the result right away
        if not self.genealogy_history:
            return genealogy
        to_prune = {}
        for node in genealogy.keys():
            if node not in self.genealogy_history:
                to_prune[node] = genealogy[node]
        for parents in genealogy.values():
            for node in parents.copy():
                if node in to_prune:
                    parents.discard(node)
                    parents.update(to_prune[node])
        for node in to_prune.keys():
            del genealogy[node]
        return genealogy
