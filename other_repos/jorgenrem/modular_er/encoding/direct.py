

"""
Direct encoding individual
"""

from enum import Enum
from gym_rem.morph.three import Rect, Servo
from collections import deque
import numpy as np


def load(toolbox, settings):
    """Load the direct encoding into the toolbox"""
    max_size = settings.getint('morphology', 'max_size')
    max_depth = settings.getint('morphology', 'max_depth')
    toolbox.register('encoding', Modular.random,
                     size=max_size, depth=max_depth,
                     history=toolbox.history)
    toolbox.register('mate', Modular.one_point_cross, history=toolbox.history)
    toolbox.register('prune', _prune_direct, max_depth=max_depth)
    # Register mutation
    morphology_prob = settings.getfloat('encoding', 'morphology_prob')
    morph_type_prob = settings.get('encoding', 'morphological_type_prob',
                                   fallback=None)
    if morph_type_prob is not None:
        morph_type_prob = eval(morph_type_prob)
    toolbox.register('mutate', Modular.mutate, prob_m=morphology_prob,
                     morph_mut_prob=morph_type_prob, history=toolbox.history)


def _prune_direct(indiv, max_depth=1000):
    """Helper function to prune a direct encoding individual"""
    # Explicit queue handling to avoid invalidating iterators
    queue = deque([indiv.morphology.root])
    while queue:
        module = queue.popleft()
        # Remove every module in the genome that is deeper than max depth
        if module.depth > max_depth:
            parent = module.parent
            conn = parent.connection_point(module)
            del parent[conn]
        else:
            queue.extend(module.children)
    return indiv


# Enumeration of possible morphological mutations
morph_mutation = Enum('morph_mutation', 'add remove mutate')


def _mutate_morphology(morphology, max_size, depth, ctrl, p=None):
    """Helper method to mutate morphology"""
    mutation = np.random.choice(list(morph_mutation), p=p)
    if mutation == morph_mutation.add:
        # Do early return if size is too large
        if len(morphology) >= max_size:
            return
        # Create a list of all possible places to insert a new module
        sites = []
        for module in morphology:
            if module.depth < depth:
                sites.extend([(site, module) for site in module.available])
        # If there are no places to put new modules then we don't do anything
        if not sites:
            return
        # Select a random site
        idx = np.random.randint(0, len(sites))
        site, module = sites[idx]
        # Select a random module type
        new_mod = np.random.choice([Rect, Servo])()
        # Insert new module
        module[site] = new_mod
        # If the newly attached module is a joint we need to attach a
        # controller
        if new_mod.joint is not None:
            # For 'add' operation we simply generate a new random controller
            new_mod.ctrl = ctrl()
    elif mutation == morph_mutation.remove:
        # If there are no modules to remove then we exit out early
        if len(morphology) <= 1:
            return
        # Select random module, all except the root are acceptable
        modules = [m for m in morphology]
        module = modules[np.random.randint(1, len(modules))]
        # Extract parent and remove module
        parent = module.parent
        del parent[module]
    elif mutation == morph_mutation.mutate:
        # Select a random module
        modules = [m for m in morphology]
        module = modules[np.random.randint(0, len(modules))]
        # Edit in place
        if isinstance(module, Servo):
            module.rotate(np.random.randint(0, 4))
        elif isinstance(module, Rect):
            module.rotate(np.random.randint(0, 4))
        else:
            raise NotImplementedError("Add support for mutating '{}'\
                    in-place".format(module))

    else:
        raise NotImplementedError("Unsupported morphological mutation: {}"
                                  .format(mutation))


class Modular(object):
    """Modular robot individual which contains a morphology, controller and
    fitness"""

    def __init__(self, morphology, control, fitness, max_size=None,
                 max_depth=1000):
        self.morphology = morphology
        self.control = control
        self.fitness = fitness
        self.max_size = max_size
        self.max_depth = max_depth
        # Morphology corresponding to what was spawned in simulation
        self.spawned_morph = None

    def reset(self):
        """Reset dynamic information"""
        self.spawned_morph = None
        del self.fitness.values

    @staticmethod
    def random(control, fitness, size, depth, history=None):
        """Create a new random modular individual with the given controller,
        fitness base and maximum size"""
        # Start by defining a root module
        root = Rect()
        # Select a random amount of modules to add
        num_modules = np.random.randint(1, size)
        for _ in range(num_modules):
            # Create a list of all possible places to insert a new module
            sites = []
            for module in root:
                if module.depth < depth:
                    sites.extend([(site, module) for site in module.available])
            # If we can't find any where to place the module with the given
            # depth restriction we give up and return the current version
            if not sites:
                break
            # Select a random site
            idx = np.random.randint(0, len(sites))
            site, module = sites[idx]
            # Select a random module type
            new_mod = np.random.choice([Rect, Servo])()
            # Insert new module
            module[site] = new_mod
            # If module is a joint type we add a controller for the joint
            if new_mod.joint is not None:
                new_mod.ctrl = control()
        # Create new individual from random tree above
        new_indiv = Modular(root, control, fitness(), size, depth)
        # Register with history
        if history is not None:
            history([new_indiv])
        # Return the newly generated modular individual
        return new_indiv

    @staticmethod
    def mutate(modular, prob_m, morph_mut_prob=None, history=None):
        """Mutate an individual"""
        # Always perform controller mutation when mutation
        for module in modular.morphology:
            if module.joint:
                module.ctrl.mutate()
        # With a certain probability also mutate morphology
        if np.random.random() <= prob_m:
            _mutate_morphology(modular.morphology,
                               modular.max_size,
                               modular.max_depth,
                               modular.control,
                               morph_mut_prob)
        # Lastly register mutated individual with history
        if history is not None:
            history([modular])
        return (modular,)

    @staticmethod
    def one_point_cross(ind_a, ind_b, history=None):
        """Crossover the two individuals by selecting a random point in both
        and exchange subtrees"""
        # Select random points in both individuals, take care to not select the
        # root
        morph_a = [m for m in ind_a.morphology if m.depth < ind_a.max_depth]
        morph_b = [m for m in ind_b.morphology if m.depth < ind_b.max_depth]
        # If there are not enough modules then do not perform crossover
        if len(morph_a) < 2 or len(morph_b) < 2:
            return ind_a, ind_b
        module_a = morph_a[np.random.randint(1, len(morph_a))]
        module_b = morph_b[np.random.randint(1, len(morph_b))]
        # Extract parents from both modules
        parent_a = module_a.parent
        parent_b = module_b.parent
        conn_a = parent_a.connection_point(module_a)
        conn_b = parent_b.connection_point(module_b)
        # Delete sub-trees from each
        del parent_a[conn_a]
        del parent_b[conn_b]
        # Crossover
        parent_a[conn_a] = module_b
        parent_b[conn_b] = module_a
        # NOTE: Since we attach controllers to each module these are also
        # copied during the crossover
        # Before returning register individuals with history object
        if history is not None:
            history([ind_a, ind_b])
        # Remove information about fitness
        ind_a.reset()
        ind_b.reset()
        # Return individuals with updated morphology
        return ind_a, ind_b
