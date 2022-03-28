import argparse
import logging
from signal import signal, SIGINT, SIGTERM

from client import Client
from model import Model
from control import SequencerControl
from utilities import get_config, get_datetime_now, DT_FORMAT, Species #TODO: Catch-all class ugly
from preprocess import SignalProcessor


# TODO: Annotate function signatures (arg types, return type)
# TODO: Comments

SAMPLING_HZ = 3012


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


def main():

    # up to nargs https://docs.python.org/3/library/argparse.html 

    # CL args
    parser.add_argument('target', # 2 options only. Compulsory.
                        help='RNA species to enrich for.')
    parser.add_argument('duration', # Compulsory
                        help='Length of time (in hours) to run RISER for. This'
                             'should be the same as your MinKNOW run length.')
    parser = argparse.ArgumentParser(description='Enrich a Nanopore sequencing run for RNA of a given species.')
    parser.add_argument('-c', '--config', # Optional, default
                        help='Config file for model hyperparameters.')
    parser.add_argument('-m', '--model', # Optional, default
                        help='File containing saved model weights.')
    parser.add_argument('-p', '--polya', # Optional, default
                        action='store_const',
                        const=6481,
                        help='Length of polyA + sequencing adapter to trim from start of signal.')
    parser.add_argument('-s', '--secs', # Optional, default.
                        action='store_const',
                        const=4,
                        help='Number of seconds of transcript signal to use for decision.') # TODO: Convert to # signal values
    args = parser.parse_args()

    # config_file = './local_data/configs/train-cnn-20.yaml'
    # model_file = 'local_data/models/train-cnn-20_0_best_model.pth'
    # polyA_length = 6481
    # input_length = 12048
    # target = Species.NONCODING
    # duration_h = 0.03

    # Set up
    out_file = f'riser_{get_datetime_now()}'
    logger = setup_logging(out_file)
    client = Client(logger)
    config = get_config(config_file)
    model = Model(model_file, config, logger)
    input_length = args.secs * SAMPLING_HZ
    processor = SignalProcessor(polyA_length, input_length)
    control = SequencerControl(client, model, processor, logger, out_file)

    # Log initial setup
    # logger.info(" ".join(sys.argv)) # TODO: Replace below with this
    logger.info('Config file: %s', config_file)
    logger.info('Model file: %s', model_file)
    logger.info('PolyA + seq adapter length: %s', polyA_length)
    logger.info('Input length: %s', input_length)
    logger.info('Target: %s', target.name)

    # Set up graceful exit
    signal(SIGINT, lambda *x: graceful_exit(control))
    signal(SIGTERM, lambda *x: graceful_exit(control))

    # Run analysis
    client.start_streaming_reads()
    control.enrich(target, duration_h)
    control.finish()


if __name__ == "__main__":
    main()
