
"""
Command line utility to extract CSV from result ZIPs
"""
from datetime import datetime
from termcolor import colored
import argparse
import configparser
import csv
import os.path
import pickle
import tqdm
import zipfile


LOGBOOK_HEADER = ['gen', 'evals', 'median', 'mean', 'std', 'min', 'max']
CSV_HEADER = ['file', 'seed', 'date', 'type', 'ea',
              'crossover', 'morph_rate', 'ctrl_sigma',
              'finished', 'duration']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSV result extractor")
    parser.add_argument('file', nargs='+', help="ZIP archives to process")
    parser.add_argument('--output', '-o', default='summary.csv',
                        help="Output of utility as CSV file")
    args = parser.parse_args()
    with open(args.output, 'w') as csvfil:
        writer = csv.DictWriter(csvfil, CSV_HEADER + LOGBOOK_HEADER)
        writer.writeheader()
        prog = tqdm.tqdm(args.file, desc="Extracting CSV", unit='archive',
                         dynamic_ncols=True)
        for fil in prog:
            file_name = os.path.splitext(os.path.basename(fil))[0]
            with zipfile.ZipFile(fil, 'r') as archive:
                # Extract configuration so that additional parameters can be
                # added to CSV output
                cfg = configparser.ConfigParser()
                cfg_exp = archive.read('experiment.ini')
                cfg.read_string(cfg_exp.decode('utf-8'), fil)
                start_time = datetime(*archive.getinfo('experiment.ini')
                                      .date_time)
                # Create list of all seeds within archive, note that this is
                # equivalent to listing all folders within the archive. We
                # filter on the word 'checkpoint' to get only the seed name out
                seeds = set(filter(lambda p: p and 'checkpoint' not in p,
                                   map(os.path.dirname, archive.namelist())))
                for seed in tqdm.tqdm(seeds, desc="Extracting seeds", unit="seed",
                                      leave=False):
                    # Create a list of only pickle files
                    pickles = [n for n in archive.namelist()
                               if n.endswith('.pickle') and seed in n]
                    # Extract log data and write to CSV
                    logs = [n for n in pickles if 'log' in n]
                    # Check if we have a log without checkpoint
                    no_check = [n for n in logs if 'checkpoint' not in n]
                    ckpts = [n for n in logs if 'checkpoint' in n]
                    # If 'no_check' is empty then we have to use checkpoints
                    use_ckpt = False
                    if not no_check:
                        prog.write(colored("Seed '{!s}' of '{!s}' did not finish"
                                           .format(seed, fil), 'yellow'))
                        if not ckpts:
                            prog.write(colored("\tNo checkpoints either...",
                                               'red'))
                            continue
                        # To ensure that checkpoints are sorted we hackily sort
                        # on 'generation'
                        no_check = sorted(ckpts, key=lambda x: int(x.split('_')[1]))
                        use_ckpt = True
                    # There will only be one log file per seed
                    log = no_check[-1]
                    logbook = pickle.loads(archive.read(log))
                    time = datetime(*archive.getinfo(log).date_time)
                    elapsed = time - start_time
                    for typ in logbook.chapters.keys():
                        data = logbook.chapters[typ].select(*LOGBOOK_HEADER)
                        for line in zip(*data):
                            row = dict(zip(LOGBOOK_HEADER, line))
                            row['file'] = file_name
                            row['seed'] = seed
                            row['date'] = start_time.isoformat(timespec='seconds')
                            row['type'] = typ
                            row['ea'] = cfg['experiment']['ea']
                            if row['ea'] == 'nsga2' and 'penalty_func' not in cfg['ea']:
                                row['ea'] = 'nsga2-norm'
                            # if row['ea'] == 'single' and 'elitism' in cfg['ea']:
                                # row['ea'] = 'single_{!s}'.format(cfg['ea']['elitism'])
                            row['crossover'] = cfg['ea']['crossover_prob']
                            row['morph_rate'] = cfg['encoding']['morphology_prob']
                            # row['ctrl_rate'] = cfg['encoding']['control_prob']
                            row['ctrl_sigma'] = cfg['control']['sigma']
                            row['finished'] = not use_ckpt
                            row['duration'] = elapsed.total_seconds()
                            writer.writerow(row)
