#!/usr/bin/env python

"""
Command line utility to extract profiling data from experiments into CSV files
"""
from termcolor import colored
import argparse
import configparser
import csv
import os.path
import pickle
import tqdm
import zipfile


LOGBOOK_HEADER = ['gen', 'name', 'duration']
CSV_HEADER = ['file', 'seed', 'ea']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSV profile extractor")
    parser.add_argument('file', nargs='+', help="ZIP archives to process")
    parser.add_argument('--output', '-o', default='profile.csv',
                        help="Output of utility as CSV file")
    args = parser.parse_args()
    with open(args.output, 'w') as csvfil:
        writer = csv.DictWriter(csvfil, CSV_HEADER + LOGBOOK_HEADER)
        writer.writeheader()
        prog = tqdm.tqdm(args.file, desc="Extracting profiling",
                         unit='archive', dynamic_ncols=True)
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
                profile = [n for n in pickles if 'profile' in n]
                if not profile:
                    ckpts = [n for n in pickles if 'checkpoint' in n]
                    if not ckpts:
                        prog.write(colored("No profile data available for '{!s}'"
                                           .format(fil), 'red'))
                        continue
                    profile = [ckpts[-1]]
                profile = profile[0]
                data = pickle.loads(archive.read(profile))
                # The following indicates that we loaded from a checkpoint, the
                # 'profile.pickle' files will simply be the data we are after
                if 'profile' in data:
                    data = data['profile']
                seed = os.path.dirname(profile)
                for row in data:
                    row = dict(zip(LOGBOOK_HEADER, row))
                    row['file'] = file_name
                    row['seed'] = seed
                    row['ea'] = cfg['experiment']['ea']
                    writer.writerow(row)
