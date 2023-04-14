

"""
Command line utility to work with 'population' saves
"""
from modular_er.ea.map_elites import Map, size_behavior
from termcolor import colored
import argparse
import configparser
import csv
import os.path
import pickle
import re
import sys
import tqdm
import zipfile


SHARED_HEADER = ['file', 'seed', 'ea', 'evaluations']


def _project_population(population, map_size=20):
    # Create new empty map and fill it with the loaded population
    m = Map((map_size, map_size))
    for indiv in population:
        behave = size_behavior(indiv, float(map_size))
        m.insert(behave, indiv)
    # Extract fitness map and output to CSV
    fitness_map = m.fitness()
    result = []
    for x in range(map_size):
        for y in range(map_size):
            result.append({'non_movable': x,
                           'movable': y,
                           'fitness': fitness_map[x, y]})
    return result


def _extract_morphology(population):
    result = []
    for indiv in population:
        result.append({'type': 'genome',
                       'count': len(indiv.morphology)})
        result.append({'type': 'morph',
                       'count': len(indiv.spawned_morph)})
        result.append({'type': 'joint',
                       'count': len([m for m in indiv.spawned_morph
                                     if m.joint])})
        result.append({'type': 'non_movable',
                       'count': len([m for m in indiv.spawned_morph
                                     if not m.joint])})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line utility to process 'population' saves")
    # Global options
    parser.add_argument('file', nargs='+', help="ZIP archive(s) to process")
    parser.add_argument('--output', '-o', required=True,
                        help="Output file to store CSV results")
    # Add subparsers for different types of processing
    subs = parser.add_subparsers(help="Type of process to transform input")
    # Add map projection command
    maps = subs.add_parser('map', help="Project population into map")
    maps.set_defaults(header=['non_movable', 'movable', 'fitness'],
                      func=_project_population)
    # Add morphology parser command
    hist = subs.add_parser('hist', help="Extract morphology information")
    hist.set_defaults(header=['type', 'count'],
                      func=_extract_morphology)
    # Parse arguments
    args = parser.parse_args()
    # Abort if output file exist
    if os.path.exists(args.output):
        sys.exit(colored("Output file '{}' already exists".format(args.output),
                         'red'))
    # Open output file for writing
    with open(args.output, 'w') as csvfil:
        # Initialize CSV output
        writer = csv.DictWriter(csvfil, SHARED_HEADER + args.header)
        writer.writeheader()
        # Create progressbar per ZIP archive
        prog = tqdm.tqdm(args.file, desc="Processing ZIPs", unit='archive')
        for fil in prog:
            # Extract filename for outputting
            file_name = os.path.splitext(os.path.basename(fil))[0]
            with zipfile.ZipFile(fil, 'r') as archive:
                # Extract configuration so that additional parameters can be
                # added to CSV output
                cfg = configparser.ConfigParser()
                cfg_exp = archive.read('experiment.ini')
                cfg.read_string(cfg_exp.decode('utf-8'), fil)
                # Create list of all seeds within archive, note that this is
                # equivalent to listing all folders within the archive. We
                # filter on the word 'checkpoint' to get only the seed name out
                seeds = set(filter(lambda p: p and 'checkpoint' not in p,
                                   map(os.path.dirname, archive.namelist())))
                for seed in tqdm.tqdm(seeds, desc="Extracting seeds",
                                      unit="seed", leave=False):
                    # Create a list of only pickle files
                    pickles = [n for n in archive.namelist() if
                               n.endswith('population.pickle') and seed in n]
                    # All archives that finished will contain 'population'
                    population = [n for n in pickles
                                  if 'checkpoint' not in n]
                    checkpoints = [n for n in pickles
                                   if 'checkpoint' in n]
                    # Warn user if no final population was found
                    if not population:
                        prog.write(colored("{!s} did not finish"
                                           .format(fil), 'yellow'))
                        if not checkpoints:
                            prog.write(colored("\tNo checkpoints either!",
                                               'red'))
                            continue
                    # Add checkpoints to population and process everything
                    population.extend(checkpoints)
                    data = []
                    # Process all populations in pool using desired function
                    for pop in population:
                        try:
                            data.append(pickle.loads(archive.read(pop)))
                        except EOFError:
                            prog.write(colored("Seed {!s} ({!s}) ran out of input".format(seed, pop), 'yellow'))
                            continue
                    for res, pop in zip(map(args.func, data), population):
                        # Write each result to CSV
                        evaluations = cfg['ea']['evaluations']
                        if 'checkpoint' in pop:
                            gen = re.search("checkpoint_([0-9]+)_([0-9]+)",
                                            pop)
                            if gen:
                                evaluations = int(gen.group(2))
                            else:
                                prog.write(colored("Broken checkpoint '{}'".format(pop), 'red'))
                                continue
                        row = {'file': file_name,
                               'seed': seed,
                               'evaluations': evaluations,
                               'ea': cfg['experiment']['ea']}
                        for result in res:
                            row.update(result)
                            writer.writerow(row)
