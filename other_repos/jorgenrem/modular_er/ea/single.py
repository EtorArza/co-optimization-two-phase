

"""
Single objective EA
"""
from deap import base, tools, algorithms
from termcolor import colored
from tqdm import tqdm
import modular_er.control
import modular_er.encoding
import modular_er.eval
import numpy as np


def _selection(toolbox, settings):
    """Helper method to extract and register selection"""
    # Setup selection algorithm
    selection = settings.get('ea', 'selection')
    if selection == 'tournament':
        tournsize = settings.getint('ea', 'tournament_size')
        toolbox.register('selection', tools.selTournament, tournsize=tournsize)
    elif selection == 'roulette':
        toolbox.register('selection', tools.selRoulette)
    else:
        raise RuntimeError("Unknown selection type: '{}'".format(selection))


# Using explicit class instead of 'creator.create' so that it can be pickled
# for MPI
class _Fitness(base.Fitness):
    weights = (1.,)


def run(toolbox, settings, initial=None, no=None):
    assert not no is None
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
    # Setup individual for evolution
    toolbox.register('individual', toolbox.encoding, toolbox.control,
                     _Fitness)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # Number of evaluations performed
    num_evals = 0
    # Define evaluation function
    modular_er.eval.load(toolbox, settings, no)
    # Setup selection algorithm
    _selection(toolbox, settings)
    if initial is None:
        # Create population
        population = toolbox.population(n=settings.getint('ea', 'population'))
    else:
        # NOTE: We always create a new population and copy just the morphology
        # from the initial population so that different 'Fitness' types do not
        # clash
        population = toolbox.population(n=len(initial))
        for new, restore in zip(population, initial):
            new.morphology = restore.morphology
    # Evaluate first random population
    fits = toolbox.map(toolbox.evaluate, population)
    num_evals += len(population)
    for ind, (fit, spawned) in zip(population, fits):
        ind.fitness.values = (fit,)
        ind.spawned_morph = spawned
    # Update history and statistics
    toolbox.history_register(0, population)
    record = multi_stat.compile(population)
    logbook.record(gen=0, evals=num_evals, **record)
    toolbox.checkpoint(0, num_evals, {'log': logbook,
                                      'population': population})
    # Setup progress bar
    total_evals = settings.getint('ea', 'evaluations')
    prog = tqdm(total=total_evals,
                initial=num_evals,
                desc="Single objective",
                disable=settings.getboolean('experiment', 'quiet'),
                unit='evaluation')
    # Extract EA settings
    cxpb = settings.getfloat('ea', 'crossover_prob')
    mutpb = settings.getfloat('ea', 'mutation_prob')
    # NOTE: 'max' ensures that negative values are discarded
    num_elites = max(0, settings.getint('ea', 'elitism', fallback=0))
    pop_size = len(population) - num_elites
    try:
        gen = 0
        while num_evals < total_evals:
            offspring = algorithms.varAnd(population, toolbox,
                                          cxpb=cxpb, mutpb=mutpb)
            # Extract all individuals that don't have a valid fitness for
            # evaluation
            invalid = [ind for ind in offspring
                       if not ind.fitness.valid]
            # Perform pruning of genomes after variation to avoid unbound
            # growth
            invalid = list(map(toolbox.prune, invalid))
            fits = toolbox.map(toolbox.evaluate, invalid)
            for ind, (fit, spawned) in zip(invalid, fits):
                ind.fitness.values = (fit,)
                ind.spawned_morph = spawned
            # Extract elite solutions from previous population, using the
            # previous population allows elites to propagate without changes to
            # control or morphology
            elites = tools.selBest(population, k=num_elites)
            # Perform selection with all offspring previously selected
            population = toolbox.selection(offspring, k=pop_size)
            population.extend(elites)
            # Update history and statistics
            gen += 1
            num_evals += len(invalid)
            prog.update(n=len(invalid))
            toolbox.history_register(gen, population)
            record = multi_stat.compile(population)
            logbook.record(gen=gen,
                           evals=len(invalid),
                           **record)
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
