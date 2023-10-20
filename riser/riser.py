import argparse
import logging
from signal import signal, SIGINT, SIGTERM
import sys
from types import SimpleNamespace

from client import Client
from model import Model
from control import SequencerControl
from utilities import get_config, get_datetime_now, DT_FORMAT #TODO: Catch-all class ugly
from preprocess import SignalProcessor


# TODO: Documentation


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


def parse_args(parser):
    args = parser.parse_args()

    # If no config specified as arg, set based on target
    if args.config is None:
        args.config = f"model/{args.target}_config_R9.4.1.yaml"

    # If no model specified as arg, set based on target
    if args.model is None:
        args.model = f"model/{args.target}_model_R9.4.1.pth"

    return args


def main():
    # CL args
    parser = argparse.ArgumentParser(description=('Enrich a Nanopore sequencing'
                                                  ' run for RNA of a given'
                                                  ' class.'))
    parser.add_argument('-t', '--target',
                        choices=['mRNA','globin'],
                        help='RNA class to enrich for. This must be either '
                             '{%(choices)s}. (required)',
                        required=True)
    parser.add_argument('-m', '--mode',
                        choices=['enrich', 'deplete'],
                        help='Whether to enrich or deplete the target class.'
                             ' (required)',
                        required=True)
    parser.add_argument('-d', '--duration',
                        dest='duration_h',
                        type=int,
                        help='Length of time (in hours) to run RISER for. '
                             'This should be the same as the MinKNOW run '
                             'length. (required)',
                        required=True)
    parser.add_argument('--config',
                        dest='config_file',
                        help='Config file for model hyperparameters. (default: '
                             '%(default)s)')
    parser.add_argument('--model',
                        dest='model_file',
                        help='File containing saved model weights. (default: '
                             '%(default)s)')
    parser.add_argument('--min',
                        default=2,
                        type=int,
                        help='Minimum number of seconds of transcript signal to'
                             ' use for decision. (default: %(default)s)')
    parser.add_argument('--max',
                        default=4,
                        type=int,
                        help='Maximum number of seconds of transcript signal to '
                            'try to classify before skipping this read. '
                            '(default: %(default)s)')
    parser.add_argument('--threshold',
                        default=0.9,
                        type=probability,
                        help='Probability threshold for classifier [0,1] '
                             '(default: %(default)s)')
    args = parse_args(parser)

    # Local testing
    # args = SimpleNamespace()
    # args.target = 'mRNA'
    # args.mode = 'deplete'
    # args.duration_h = 0.05
    # args.config_file = 'riser/model/mRNA_config_R9.4.1.yaml'
    # args.model_file = 'riser/model/mRNA_model_R9.4.1.pth'
    # args.min = 2
    # args.max = 4
    # args.threshold = 0.9

    # Set up
    out_file = f'riser_{get_datetime_now()}'
    logger = setup_logging(out_file)
    client = Client(logger)
    config = get_config(args.config_file)
    model = Model(args.model_file, config, logger)
    processor = SignalProcessor(args.trim_length, args.min, args.max)
    control = SequencerControl(client, model, processor, logger, out_file)

    # Log CL args
    logger.info(f'Usage: {" ".join(sys.argv)}')
    logger.info('All settings used (including those set by default):')
    for k,v in vars(args).items(): logger.info(f'--{k:14}: {v}')

    # Set up graceful exit
    signal(SIGINT, lambda *x: graceful_exit(control))
    signal(SIGTERM, lambda *x: graceful_exit(control))

    # Run analysis
    control.start()
    control.target(args.mode, args.duration_h, args.threshold)
    control.finish()


if __name__ == "__main__":
    main()
