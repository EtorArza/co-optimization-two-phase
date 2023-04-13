#!/usr/bin/env python
"""
Command line utility to extract Histogram data from populations
"""
from termcolor import colored
import argparse
import configparser
import csv
import os.path
import pickle
import re
import tqdm
import zipfile


CSV_HEADER = ['file', 'seed', 'ea', 'type', 'generation', 'count']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Histogram extractor")
    parser.add_argument('file', nargs='+', help="ZIP archives to process")
    parser.add_argument('--output', '-o', default='histogram.csv',
                        help="Output of utility as CSV file")
    args = parser.parse_args()
    with open(args.output, 'w') as csvfil:
        writer = csv.DictWriter(csvfil, CSV_HEADER)
        writer.writeheader()
        prog = tqdm.tqdm(args.file, desc='Creating histogram', unit='archive')
        for fil in prog:
            file_name = os.path.splitext(os.path.basename(fil))[0]
            with zipfile.ZipFile(fil, 'r') as archive:
                # Extract configuration so that additional parameters can be
                # added to CSV output
                cfg = configparser.ConfigParser()
                cfg_exp = archive.read('experiment.ini')
                cfg.read_string(cfg_exp.decode('utf-8'), fil)
                # Create a list of only pickle files
                pickles = [n for n in archive.namelist()
                           if n.endswith('.pickle')]
                # Extract log data and write to CSV
                logs = [n for n in pickles if 'population' in n]
                if not logs:
                    prog.write(colored("'{!s}' did not finish"
                                       .format(fil), 'yellow'))
                checkpoints = [n for n in pickles if 'checkpoint' in n]
                logs.extend(checkpoints)
                prog2 = tqdm.tqdm(logs,
                                  desc="Processing '{!s}'".format(file_name),
                                  unit="population", leave=False)
                for log in prog2:
                    seed = os.path.dirname(log)
                    generation = -1
                    if 'population' in log:
                        # Explicit cast to list for MAP-Elites
                        population = list(pickle.loads(archive.read(log)))
                        generation = cfg['ea']['generations']
                    elif 'checkpoint' in log:
                        data = pickle.loads(archive.read(log))
                        if 'population' in data:
                            population = list(data['population'])
                            gen = re.search("checkpoint_([0-9]+).pickle", log)
                            if gen:
                                generation = int(gen.group(1))
                        else:
                            continue
                    row = {'file': file_name,
                           'seed': seed,
                           'generation': generation,
                           'ea': cfg['experiment']['ea']}
                    if row['ea'] == 'nsga2' and 'penalty_func' not in cfg['ea']:
                        row['ea'] = 'nsga2-norm'
                    pop_lens = []
                    for indiv in population:
                        row['type'] = 'genome'
                        row['count'] = len(indiv.morphology)
                        writer.writerow(row)
                        row['type'] = 'morph'
                        row['count'] = len(indiv.spawned_morph)
                        writer.writerow(row)
                        row['type'] = 'joint'
                        joint_count = len([m for m in indiv.spawned_morph
                                           if m.joint])
                        row['count'] = joint_count
                        writer.writerow(row)
                        row['type'] = 'non_movable'
                        non_count = len([m for m in indiv.spawned_morph
                                         if not m.joint])
                        row['count'] = non_count
                        writer.writerow(row)
                        row['type'] = 'module_ratio'
                        row['count'] = float(joint_count) / float(non_count)
                        writer.writerow(row)
