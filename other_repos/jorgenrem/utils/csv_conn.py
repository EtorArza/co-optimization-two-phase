

"""
Command line utility to extract genealogy connectivity information into CSV
"""
from termcolor import colored
import argparse
import configparser
import csv
import networkx
import numpy as np
import os.path
import pickle
import re
import tqdm
import zipfile


CSV_HEADER = ['file', 'seed', 'ea', 'evaluation', 'children', 'parents',
              'parent_diff']


def _compose_population(history, population):
    """This method creates a 'networkx' graph, combining the ancestry of all
    individuals in the population"""
    graphs = []
    for ind in population:
        genes = history.genealogy(ind)
        net = history.network(genes, prune=True)
        if net:
            graphs.append(net)
    return networkx.algorithms.operators.all.compose_all(graphs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genealogy connectivity")
    parser.add_argument('file', nargs='+', help="ZIP archive to process")
    parser.add_argument('--output', '-o', default='connectivity.csv',
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
                        prog2.set_description("Composing population for {!s}".format(seed))
                        net = _compose_population(history, data)
                        for node in net.nodes:
                            row['children'] = net.in_degree(node)
                            row['parents'] = net.out_degree(node)
                            diff = []
                            info = net.nodes[node]
                            for parent in net.successors(node):
                                p_info = net.nodes[parent]
                                diff.append(info['fitness'] - p_info['fitness'])
                            if diff:
                                row['parent_diff'] = np.average(diff)
                            else:
                                row['parent_diff'] = -1
                            writer.writerow(row)
