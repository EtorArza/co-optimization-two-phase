#!/usr/bin/env python

"""
MAP-Elites implementation for DEAP and modular robots
"""
from deap import base, tools, algorithms
from termcolor import colored
from tqdm import tqdm
import math
import modular_er.control
import modular_er.encoding
import modular_er.eval
import numpy as np


class Map(object):
    """Storage container for MAP-Elites"""

    def __init__(self, size):
        self._size = tuple(size)
        self._storage = [[None for _ in range(size[1])]
                         for _ in range(size[0])]

    def coord(self, key):
        """Calculate coordinate in underlying storage based on key"""
        assert all(map(lambda k: 0 <= k <= 1, key)), "Key '{}' is outside\
                range [0, 1]".format(key)
        # NOTE: We subtract '1' since we use 'range' when creating which will
        # give us a sizes in [0, size)
        key = [math.ceil(k * (s - 1)) for k, s in zip(key, self._size)]
        return tuple(key)

    def __iter__(self):
        for x in range(self._size[0]):
            for y in range(self._size[1]):
                item = self._storage[x][y]
                if item:
                    yield item

    def insert(self, key, item):
        # Assume key is [0, 1]
        x, y = self.coord(key)
        # If the map contains no solution with the given behavior
        if not self._storage[x][y]:
            self._storage[x][y] = item
            return True
        if self._storage[x][y].fitness < item.fitness:
            self._storage[x][y] = item
            return True
        return False

    def __getitem__(self, key):
        # Assume key is [0, 1]
        x, y = self.coord(key)
        return self._storage[x][y]

    def __len__(self):
        length = 0
        for x in range(self._size[0]):
            for y in range(self._size[1]):
                item = self._storage[x][y]
                if item:
                    length += 1
        return length

    def fitness(self):
        """Return the map with only fitness values"""
        result = np.zeros(self._size)
        for x in range(self._size[0]):
            for y in range(self._size[1]):
                item = self._storage[x][y]
                if item:
                    result[x, y] = item.fitness.values[0] + 0.001
        return result


# Using explicit class instead of 'creator.create' so that it can be pickled
# for MPI
class _Fitness(base.Fitness):
    # Weights used by DEAP to maximize or minimize
    weights = (1.,)
    # Behavior characteristics used for placement in map
    behavior = None
    # Curiosity score, used for selection when 'best' or 'tournament' is
    # selected
    curiosity = 0.0


def size_behavior(ind, max_size=1.):
    """Calculate behavior for a given individual"""
    non_movable = len([m for m in ind.spawned_morph if not m.joint])
    movable = len([m for m in ind.spawned_morph if m.joint])
    # Since root is always present
    non_norm = (non_movable - 1.) / max_size
    mov_norm = movable / max_size
    return non_norm, mov_norm


def run(toolbox, settings, initial=None):
    """Run a full evolutionary run for MAP-Elites"""
    # Create statistics and logging
    fit_stat = tools.Statistics(lambda ind: ind.fitness.values[0])
    gen_len = tools.Statistics(lambda ind: len(ind.morphology))
    morph_len = tools.Statistics(lambda ind: len(ind.spawned_morph))
    joint_len = tools.Statistics(lambda ind: len([m for m in ind.spawned_morph
                                                  if m.joint]))
    static_len = tools.Statistics(lambda ind: len([m for m in ind.spawned_morph
                                                   if not m.joint]))
    multi_stat = tools.MultiStatistics(fitness=fit_stat,
                                       genome=gen_len,
                                       morph=morph_len,
                                       joint=joint_len,
                                       non_movable=static_len)
    multi_stat.register('median', np.median)
    multi_stat.register('mean', np.mean)
    multi_stat.register('std', np.std)
    multi_stat.register('min', np.min)
    multi_stat.register('max', np.max)
    logbook = tools.Logbook()
    # Load controller type
    modular_er.control.load(toolbox, settings)
    # Load individual encoding
    modular_er.encoding.load(toolbox, settings)
    # Setup individual for population creation
    toolbox.register('individual', toolbox.encoding, toolbox.control,
                     _Fitness)
    # NOTE: The population is only used to generate the first initial random
    # population
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # Define evaluation function
    modular_er.eval.load(toolbox, settings)
    # Register selection function
    batch_size = settings.getint('ea', 'batch_size')
    if 'selection' in settings['ea']:
        if settings['ea']['selection'] == 'best':
            toolbox.register('selection', tools.selBest,
                             k=batch_size, fit_attr='fitness.curiosity')
        elif settings['ea']['selection'] == 'tournament':
            size = settings.getint('ea', 'tournament_size', fallback=5)
            toolbox.register('selection', tools.selTournament,
                             k=batch_size, tournsize=size,
                             fit_attr='fitness.curiosity')
    else:
        toolbox.register('selection', tools.selRandom,
                         k=batch_size)
    # Register behavior function
    size = settings.getint('morphology', 'max_size')
    toolbox.register('behavior', size_behavior, max_size=float(size))
    if initial is None:
        # Create initial random population
        population = toolbox.population(n=settings.getint('ea', 'initial_size'))
    else:
        # NOTE: We always create a new population and copy just the morphology
        # from the initial population so that different 'Fitness' types do not
        # clash
        population = toolbox.population(n=len(initial))
        for new, restore in zip(population, initial):
            new.morphology = restore.morphology
    # Number of evaluations performed
    num_evals = 0
    # Create map for storage of elites
    map_pop = Map((size, size))
    # Evaluate first random population
    fits = toolbox.map(toolbox.evaluate, population)
    num_evals += len(population)
    for ind, (fit, spawned) in zip(population, fits):
        # Set fitness
        ind.fitness.values = (fit,)
        # Update spawned morphology
        ind.spawned_morph = spawned
        # Set behavior characteristics
        ind.fitness.behavior = toolbox.behavior(ind)
    # Insert into map
    for ind in population:
        map_pop.insert(ind.fitness.behavior, ind)
    # Update history and statistics
    toolbox.history_register(0, map_pop)
    record = multi_stat.compile(map_pop)
    logbook.record(gen=0, evals=len(population), **record)
    toolbox.checkpoint(0, num_evals, {'log': logbook,
                                      'population': map_pop})
    # Setup progress bar
    total_evals = settings.getint('ea', 'evaluations')
    prog = tqdm(total=total_evals,
                initial=num_evals,
                desc="MAP-Elites",
                disable=settings.getboolean('experiment', 'quiet'),
                unit='evaluation')
    cxpb = settings.getfloat('ea', 'crossover_prob')
    mutpb = settings.getfloat('ea', 'mutation_prob')
    try:
        gen = 0
        while num_evals < total_evals:
            # Select random individuals to mutate and try to insert
            parents = toolbox.selection(list(map_pop))
            # Apply variation
            batch = algorithms.varAnd(parents, toolbox,
                                      cxpb=cxpb, mutpb=mutpb)
            child_parent = {ind: par for ind, par in zip(batch, parents)}
            # There is no need for evaluating and trying to insert individuals
            # that have already been evaluated
            batch = [ind for ind in batch if not ind.fitness.valid]
            # Perform pruning of genomes after variation to avoid unbound
            # growth
            batch = list(map(toolbox.prune, batch))
            fits = toolbox.map(toolbox.evaluate, batch)
            for ind, (fit, spawned) in zip(batch, fits):
                ind.fitness.values = (fit,)
                ind.spawned_morph = spawned
                # Set behavior characteristics
                ind.fitness.behavior = toolbox.behavior(ind)
            # Try to insert into map
            for ind in batch:
                parent = child_parent[ind]
                if map_pop.insert(ind.fitness.behavior, ind):
                    parent.fitness.curiosity += 1.0
                else:
                    parent.fitness.curiosity -= 0.5
            # Update history and statistics
            gen += 1
            num_evals += len(batch)
            prog.update(n=len(batch))
            toolbox.history_register(gen, map_pop)
            record = multi_stat.compile(map_pop)
            logbook.record(gen=gen,
                           evals=len(batch),
                           **record)
            toolbox.checkpoint(gen, num_evals, {'log': logbook,
                                                'population': map_pop})
    except KeyboardInterrupt:
        # Catch and ignore so that we can return values for storage
        pass
    except Exception as e:
        import traceback
        prog.write(colored("Caught exception: {!s}".format(e), 'red'))
        prog.write(colored(traceback.format_exc(), 'yellow'))
    return {'log': logbook, 'population': map_pop}
