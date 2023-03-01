#!/usr/bin/env python
"""
Command line utility to create CSV maps from result ZIPs
"""
from termcolor import colored
from modular_er.ea.map_elites import Map, size_behavior
import argparse
import configparser
import csv
import os.path
import pickle
import re
import tqdm
import zipfile


CSV_HEADER = ['file', 'seed', 'ea', 'generation', 'non_movable', 'movable',
              'fitness', 'evaluations']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSV map extractor")
    parser.add_argument('file', nargs='+', help="ZIP archives to process")
    parser.add_argument('--output', '-o', default='maps.csv',
                        help="Output of utility as CSV file")
    args = parser.parse_args()
    with open(args.output, 'w') as csvfil:
        writer = csv.DictWriter(csvfil, CSV_HEADER)
        writer.writeheader()
        prog = tqdm.tqdm(args.file, desc="Creating Maps", unit='archive',
                         dynamic_ncols=True)
        for fil in prog:
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
                for seed in tqdm.tqdm(seeds, desc="Extracting seeds", unit="seed",
                                      leave=False):
                    # Create a list of only pickle files
                    pickles = [n for n in archive.namelist()
                               if n.endswith('population.pickle') and seed in n]
                    # All archives that finished will contain 'population'
                    population = [n for n in pickles
                                  if not 'checkpoint' in n]
                    checkpoints = [n for n in pickles
                                  if 'checkpoint' in n]
                    # Warn user if no final population was found
                    if not population:
                        prog.write(colored("{!s} did not finish"
                                           .format(fil), 'yellow'))
                        if not checkpoints:
                            prog.write(colored("\tNo checkpoints either!", 'red'))
                            continue
                    # Add checkpoints to population and process everything
                    population.extend(checkpoints)
                    for pop in population:
                        generation = -1
                        evaluations = cfg['ea']['evaluations']
                        data = pickle.loads(archive.read(pop))
                        if 'checkpoint' in pop:
                            gen = re.search("checkpoint_([0-9]+)_([0-9]+)", pop)
                            if gen:
                                generation = int(gen.group(1))
                                evaluations = int(gen.group(2))
                            else:
                                prog.write(colored("Broken checkpoint '{}'".format(pop), 'red'))
                                continue
                        # Create a new empty map and fill it with the population
                        m = Map((20, 20))
                        for indiv in data:
                            behave = size_behavior(indiv, 20.)
                            m.insert(behave, indiv)
                        # Extract fitness map and output to CSV
                        fitness_map = m.fitness()
                        # For each point in output map create a row in CSV
                        row = {'file': file_name, 'seed': seed,
                               'generation': generation,
                               'evaluations': evaluations,
                               'ea': cfg['experiment']['ea']}
                        # Check EA name for special case
                        if row['ea'] == 'nsga2' and 'penalty_func' not in cfg['ea']:
                            row['ea'] = 'nsga2-norm'
                        for x in range(20):
                            for y in range(20):
                                row['non_movable'] = x
                                row['movable'] = y
                                row['fitness'] = fitness_map[x, y]
                                writer.writerow(row)
