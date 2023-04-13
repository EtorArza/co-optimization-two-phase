#!/usr/bin/env python

"""
Command line utility to extract genealogy information into CSV
"""
from termcolor import colored
import argparse
import configparser
import csv
import numpy as np
import os.path
import pickle
import re
import tqdm
import zipfile


CSV_HEADER = ['file', 'seed', 'ea', 'evaluation', 'movable', 'non_movable',
              'fitness', 'num_parents', 'coverage', 'qd_score', 'inserted',
              'parent_diff', 'children', 'parents']


def _preprocess_node_rec(network, node):
    info = network.nodes[node]
    if 'parents' not in info:
        info['parents'] = 0
        info['parent_diff'] = -1
        diff = []
        info['map'] = {}
        info['map'][(info['movable'], info['non_movable'])] = info['fitness']
        for parent in network.successors(node):
            p_info = network.nodes[parent]
            if 'parents' not in p_info:
                _preprocess_node_rec(network, parent)
            # Add grandparents and our own parent to count
            info['parents'] += p_info['parents'] + 1
            diff.append(info['fitness'] - p_info['fitness'])
            # Iterate 'map' of parent and potentially add to child 'map'
            for key in p_info['map'].keys():
                if key not in info['map'] or info['map'][key] < p_info['map'][key]:
                    info['map'][key] = p_info['map'][key]
        if diff:
            info['parent_diff'] = np.average(diff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genealogy extractor")
    parser.add_argument('file', nargs='+', help="ZIP archive to process")
    parser.add_argument('--output', '-o', default='genealogy.csv',
                        help="Output of utility as CSV file")
    args = parser.parse_args()
    with open(args.output, 'w') as csvfil:
        writer = csv.DictWriter(csvfil, CSV_HEADER)
        writer.writeheader()
        prog = tqdm.tqdm(args.file, desc="Extracting Network data",
                         unit='archive')
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
                # Extract all histories from archive
                histories = [f for f in archive.namelist() if
                             f.endswith('history.pickle')]
                for seed in tqdm.tqdm(seeds, desc="Extracting seeds",
                                      unit="seed", leave=False):
                    history = [h for h in histories if seed in h]
                    if not history:
                        prog.write(colored("No valid history for seed '{!s}'"
                                           .format(seed),
                                           'yellow'))
                        continue
                    # Load history information
                    history = pickle.loads(archive.read(history[0]))
                    # Extract genealogy for entire archive
                    network = history.network(prune=True)
                    if not network:
                        prog.write(colored("Invalid history discovered for'{!s}'".format(file_name), 'red'))
                        continue
                    for node in tqdm.tqdm(network.nodes, desc="Preprocessing network",
                                          unit="node", leave=False):
                        _preprocess_node_rec(network, node)
                    # Extract all checkpoints containing population data
                    # NOTE: We create the progress bar first so that user gets
                    # feedback that we are working on the network since it
                    # takes a lot of time
                    pops = [f for f in archive.namelist() if
                            f.endswith("population.pickle") and seed in f]
                    prog2 = tqdm.tqdm(pops,
                                      desc="Extracting {!s}".format(seed),
                                      unit="population", leave=False)
                    for pop in prog2:
                        data = pickle.loads(archive.read(pop))
                        evaluation = cfg['ea']['evaluations']
                        if 'checkpoint' in pop:
                            gen = re.search("checkpoint_([0-9]+)_([0-9]+)",
                                            pop)
                            if gen:
                                evaluation = int(gen.group(2))
                            else:
                                prog.write(colored("Broken checkpoint '{}'".format(pop), 'red'))
                                continue
                        row = {'file': file_name, 'seed': seed,
                               'evaluation': evaluation,
                               'ea': cfg['experiment']['ea']}
                        inserted_initial = 200
                        if cfg['experiment']['ea'] == 'map-elites':
                            inserted_initial = 1000
                        for indiv in data:
                            info = network.nodes[indiv.history_index]
                            row['movable'] = info['movable']
                            row['non_movable'] = info['non_movable']
                            row['fitness'] = info['fitness']
                            row['num_parents'] = info['parents']
                            row['coverage'] = len(info['map'].keys())
                            row['qd_score'] = sum([p for p in
                                                   info['map'].values()])
                            row['inserted'] = info['generation'] * 200 + inserted_initial
                            row['parent_diff'] = info['parent_diff']
                            row['children'] = network.in_degree(indiv.history_index)
                            row['parents'] = network.out_degree(indiv.history_index)
                            writer.writerow(row)
