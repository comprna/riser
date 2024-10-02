import argparse
from collections import UserDict
from datetime import datetime
import logging
from signal import signal, SIGINT, SIGTERM
import sys
from types import SimpleNamespace

import attridict
import yaml

from client import Client
from model import Model
from control import SequencerControl
from preprocess import SignalProcessor, Kit


DT_FORMAT = '%Y-%m-%dT%H:%M:%S'


def get_config(filepath):
    with open(filepath) as config_file:
        return attridict(yaml.load(config_file, Loader=yaml.Loader))


def get_pore_version(kit):
    if kit == "RNA002":
        return "R9.4.1"
    elif kit == "RNA004":
        return "RP4"
    else:
        raise Exception(f"Invalid kit {kit}")


def get_models(targets, logger, kit):
    pore = get_pore_version(kit)
    models = []
    for target in targets:
        config = get_config(f"model/{target}_config_{kit}_{pore}.yaml")
        model_file = f"model/{target}_model_{kit}_{pore}.pth"
        models.append(Model(model_file, config, logger, target))
    return models


def get_datetime_now():
    return datetime.now().strftime(DT_FORMAT)


def setup_logging(out_file):
    logging.basicConfig(filename=f'{out_file}.log',
                        level=logging.DEBUG,
                        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                        datefmt=DT_FORMAT)

    # Also write INFO-level or higher messages to sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Turn off ReadUntil logging, which clogs up the logs
    logging.getLogger("ReadUntil").disabled = True

    return logging.getLogger("RISER")


def graceful_exit(control):
    control.finish()
    exit(0)


def probability(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a float")
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError(f"{x} not in range [0,1]")
    return x


def main():
    # CL args
    parser = argparse.ArgumentParser(description=('Enrich a Nanopore sequencing'
                                                  ' run for RNA of a given'
                                                  ' class.'))
    parser.add_argument('-t', '--target',
                        choices=['mRNA', 'globin', 'mtRNA'],
                        help='RNA class(es) to target for enrichment or '
                             'depletion. Select one or more. (required)',
                        nargs='+',
                        required=True)
    parser.add_argument('-m', '--mode',
                        choices=['enrich', 'deplete'],
                        help='Whether to enrich or deplete the target class(es).'
                             ' (required)',
                        required=True)
    parser.add_argument('-d', '--duration',
                        dest='duration_h',
                        type=float,
                        help='Length of time (in hours) to run RISER for. '
                             'This should be the same as the MinKNOW run '
                             'length. (required)',
                        required=True)
    parser.add_argument('-k', '--kit',
                        choices=['RNA002', 'RNA004'],
                        help='Sequencing kit. (required)',
                        required=True)
    parser.add_argument('-p', '--prob_threshold',
                        default=0.9,
                        type=probability,
                        help='Probability threshold for classifier [0,1] '
                             '(default: %(default)s)')
    args = parser.parse_args()

    # Local testing
    # args = SimpleNamespace()
    # args.target = ['mRNA', 'mtRNA']
    # args.mode = 'deplete'
    # args.duration_h = 0.05
    # args.kit = "RNA002"
    # args.prob_threshold = 0.9

    # Set up
    out_file = f'riser_{get_datetime_now()}'
    logger = setup_logging(out_file)
    client = Client(logger)
    models = get_models(args.target, logger, args.kit)
    kit = Kit.create_from_version(args.kit)
    processor = SignalProcessor(kit)
    control = SequencerControl(client, models, processor, logger, out_file)

    # Log CL args
    logger.info(f'Usage: {" ".join(sys.argv)}')
    logger.info('All settings used (including those set by default):')
    for k,v in vars(args).items(): logger.info(f'--{k:14}: {v}')

    # Set up graceful exit
    signal(SIGINT, lambda *x: graceful_exit(control))
    signal(SIGTERM, lambda *x: graceful_exit(control))

    # Run analysis
    control.start()
    control.target(args.mode, args.duration_h, args.prob_threshold)
    control.finish()


if __name__ == "__main__":
    main()
