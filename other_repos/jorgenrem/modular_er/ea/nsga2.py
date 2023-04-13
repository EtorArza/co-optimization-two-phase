#!/usr/bin/env python

"""
Multi-objective EA using NSGAII
"""
from deap import base, tools, algorithms
from termcolor import colored
from tqdm import tqdm
import modular_er.control
import modular_er.encoding
import modular_er.eval
import numpy as np


# Using explicit class instead of 'creator.create' so that it can be pickled
# for MPI
class _Fitness3D(base.Fitness):
    weights = (1., 1., 1.)


class _Fitness2D(base.Fitness):
    weights = (1., 1.)


def diversity(pop, dist_func):
    """Apply a distance measure to create a fitness tuple"""
    for ind1 in pop:
        distance = None
        for ind2 in pop:
            if ind1 != ind2:
                dist = dist_func(ind1, ind2)
                if distance is None:
                    distance = dist
                else:
                    distance += dist
        distance /= float(len(pop) - 1.0)
        fitness = ind1.fitness.values
        ind1.fitness.values = np.concatenate([fitness, distance])


def modules_joints_distance(ind1, ind2, func=None):
    """Calculate the distance between two individuals on number of modules and
    number of joints"""
    # Extract attributes
    ind1_mods = len([m for m in ind1.spawned_morph if not m.joint])
    ind1_joints = len([m for m in ind1.spawned_morph if m.joint])
    ind2_mods = len([m for m in ind2.spawned_morph if not m.joint])
    ind2_joints = len([m for m in ind2.spawned_morph if m.joint])
    # Create vectors, note we force the type since `len` returns int
    ind1_vec = np.array([ind1_mods, ind1_joints], dtype=np.float64)
    ind2_vec = np.array([ind2_mods, ind2_joints], dtype=np.float64)
    dist = np.abs(ind1_vec - ind2_vec)
    if func is not None:
        return func(dist)
    return dist


def modules_joints_euclid(ind1, ind2, func=None):
    """Euclidean distance between two individuals"""
    vec = modules_joints_distance(ind1, ind2, func)
    return np.array([np.linalg.norm(vec)])


def normal_one(dist):
    """Simple normalizer based on exponential function without penalty for
    equality"""
    return 1. - np.exp(-dist)


def normal_penalty(dist, penalty=0.5):
    """Normalizer function with penalty"""
    return 1. - np.exp(penalty - dist)


def run(toolbox, settings, initial=None):
    """Run a full evolutionary run of a single objective EA"""
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
    # Load controller into toolbox
    modular_er.control.load(toolbox, settings)
    # Load individual encoding
    modular_er.encoding.load(toolbox, settings)
    distance_metric = settings.get('ea', 'distance')
    penalty_func = None
    if 'penalty_func' in settings['ea']:
        if settings['ea']['penalty_func'] == 'exp':
            penalty_func = normal_one
        elif settings['ea']['penalty_func'] == 'penalty':
            penalty_func = normal_penalty
        else:
            raise NotImplementedError("Unknown penalty function: '{!s}'"
                                      .format(settings['ea']['penalty_func']))
    if distance_metric == 'distance':
        toolbox.register('fit', _Fitness3D)
        toolbox.register('distance', modules_joints_distance, func=penalty_func)
    elif distance_metric == 'euclidean':
        toolbox.register('fit', _Fitness2D)
        toolbox.register('distance', modules_joints_euclid, func=penalty_func)
    else:
        raise NotImplementedError("Not a distance measure: '{!s}'"
                                  .format(distance_metric))
    toolbox.register('diversity', diversity, dist_func=toolbox.distance)
    # Setup individual for evolution
    toolbox.register('individual', toolbox.encoding, toolbox.control,
                     toolbox.fit)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # Define evaluation function
    modular_er.eval.load(toolbox, settings, no)
    # Setup selection algorithm
    toolbox.register('selection', tools.selNSGA2)
    population_size = settings.getint('ea', 'population')
    if initial is None:
        # Create population
        assert population_size % 4 == 0, "NSGAII requires population size multiple of 4, was: {:d}".format(population_size)
        population = toolbox.population(n=population_size)
    else:
        # Check if the number of solutions are a multiple of 4 if not we remove
        # the worst performers to get multiple of 4
        mod = len(initial) % 4
        if mod != 0:
            num = len(initial) - mod
            initial = sorted(initial, key=lambda x: x.fitness, reverse=True)[:num]
        assert population_size % 4 == 0, "NSGAII requires population size multiple of 4, was: {:d}".format(population_size)
        # NOTE: We always create a new population and copy just the morphology
        # from the initial population so that different 'Fitness' types do not
        # clash
        population = toolbox.population(n=len(initial))
        for new, restore in zip(population, initial):
            new.morphology = restore.morphology
    # Number of evaluations performed
    num_evals = 0
    # Evaluate first random population
    fits = toolbox.map(toolbox.evaluate, population)
    num_evals += len(population)
    for ind, (fit, spawned) in zip(population, fits):
        ind.fitness.values = (fit,)
        ind.spawned_morph = spawned
    # Calculate diversity metrics for multi-objective
    toolbox.diversity(population)
    # This next selection does not actually perform selection, but instead
    # ensures that we assign crowding distance to all individuals
    population = toolbox.selection(population, len(population))
    # Store statistics for initial population
    toolbox.history_register(0, population)
    record = multi_stat.compile(population)
    logbook.record(gen=0, evals=num_evals, **record)
    toolbox.checkpoint(0, num_evals, {'log': logbook,
                                      'population': population})
    # Setup progressbar
    total_evals = settings.getint('ea', 'evaluations')
    prog = tqdm(total=total_evals,
                initial=num_evals,
                desc="NSGAII",
                disable=settings.getboolean('experiment', 'quiet'),
                unit='evaluation')
    cxpb = settings.getfloat('ea', 'crossover_prob')
    mutpb = settings.getfloat('ea', 'mutation_prob')
    try:
        gen = 0
        while num_evals < total_evals:
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = algorithms.varAnd(offspring, toolbox,
                                          cxpb=cxpb, mutpb=mutpb)
            # Extract all individuals that don't have a valid fitness
            invalid = [ind for ind in offspring
                       if not ind.fitness.valid]
            # Perform pruning of genomes after variation to avoid unbound
            # growth
            invalid = list(map(toolbox.prune, invalid))
            fits = toolbox.map(toolbox.evaluate, invalid)
            for ind, (fit, spawned) in zip(invalid, fits):
                ind.fitness.values = (fit,)
                ind.spawned_morph = spawned
            # Calculate diversity metrics for multi-objective
            toolbox.diversity(population + offspring)
            # Perform selection for next generation
            population = toolbox.selection(population + offspring,
                                           k=population_size)
            # Update history and statistics
            gen += 1
            num_evals += len(invalid)
            prog.update(n=len(invalid))
            toolbox.history_register(gen, population)
            record = multi_stat.compile(population)
            logbook.record(gen=gen, evals=len(invalid), **record)
            toolbox.checkpoint(gen, num_evals, {'log': logbook,
                                                'population': population})
    except KeyboardInterrupt:
        # Catch and ignore so that we can return values for storage
        pass
    except Exception as e:
        import traceback
        prog.write(colored("Caught exception: {!s}".format(e), 'red'))
        prog.write(colored(traceback.format_exc(), 'yellow'))
    return {'log': logbook, 'population': population}
