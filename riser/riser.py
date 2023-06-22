import argparse
import logging
from signal import signal, SIGINT, SIGTERM
import sys
from types import SimpleNamespace

from client import Client
from model import Model
from control import SequencerControl
from utilities import get_config, get_datetime_now, DT_FORMAT, Species #TODO: Catch-all class ugly
from preprocess import SignalProcessor


# TODO: Annotate function signatures (arg types, return type)
# TODO: Comments


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


class TargetAction(argparse.Action):
    """
    Argparse action for handling Target
    """
    def __init__(self, **kwargs):
        super(TargetAction, self).__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        value = Species.CODING if values == 'coding' else Species.NONCODING
        setattr(namespace, self.dest, value)

def main():
    # CL args
    parser = argparse.ArgumentParser(description=('Enrich a Nanopore sequencing'
                                                  ' run for RNA of a given'
                                                  ' species.'))
    parser.add_argument('-t', '--target',
                        choices=['coding', 'noncoding'],
                        action=TargetAction,
                        help='RNA species to enrich for. This must be either '
                             '{%(choices)s}. (required)',
                        required=True,
                        metavar='')
    parser.add_argument('-d', '--duration',
                        dest='duration_h',
                        type=int,
                        help='Length of time (in hours) to run RISER for. '
                             'This should be the same as the MinKNOW run '
                             'length. (required)',
                        required=True,
                        metavar='')
    parser.add_argument('-c', '--config',
                        dest='config_file',
                        default='model/cnn_best_model.yaml',
                        help='Config file for model hyperparameters. (default: '
                             '%(default)s)',
                        metavar='')
    parser.add_argument('-m', '--model',
                        dest='model_file',
                        default='model/cnn_best_model.pth',
                        help='File containing saved model weights. (default: '
                             '%(default)s)',
                        metavar='')
    parser.add_argument('-p', '--polya',
                        dest='polyA_length',
                        default=6481,
                        type=int,
                        help='Number of values to remove from the start of the '
                             'raw signal to exclude the polyA tail and '
                             'sequencing adapter signal from analysis. '
                             '(default: %(default)s)',
                        metavar='')
    parser.add_argument('-s', '--secs',
                        default=4,
                        type=int,
                        choices=range(1,10),
                        help='Number of seconds of transcript signal to use '
                             'for decision. (default: %(default)s)',
                        metavar='')
    args = parser.parse_args()

    # Local testing
    # args = SimpleNamespace()
    # args.target = Species.NONCODING
    # args.duration_h = 1
    # args.config_file = 'models/cnn_best_model.yaml'
    # args.model_file = 'models/cnn_best_model.pth'
    # args.polyA_length = 6481
    # args.secs = 4

    # Set up
    out_file = f'riser_{get_datetime_now()}'
    logger = setup_logging(out_file)
    client = Client(logger)
    config = get_config(args.config_file)
    model = Model(args.model_file, config, logger)
    processor = SignalProcessor(args.polyA_length, args.secs)
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
    control.enrich(args.target, args.duration_h)
    control.finish()


if __name__ == "__main__":
    main()