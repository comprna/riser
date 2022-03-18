from enum import Enum
from datetime import datetime
import logging
from timeit import default_timer as timer

from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
import numpy as np
import torch

from client import Client
from model import Model
from riser import Riser
from utilities import get_config
from preprocess import SignalProcessor


# TODO: Annotate function signatures (arg types, return type)
# TODO: Comments
# TODO: Extract model class to encapsulate PyTorch code

DT_FORMAT = '%Y-%m-%dT%H:%M:%S'


class Target(Enum):
    NONCODING = 0
    CODING = 1


# TODO: Duplicated in riser.py
def get_datetime_now():
    return datetime.now().strftime(DT_FORMAT)


def setup_logging():
    logging.basicConfig(filename=f'riser_{get_datetime_now()}.log',
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


def main():
    # CL args
    config_file = './local_data/configs/train-cnn-20.yaml'
    model_file = 'local_data/models/train-cnn-20_0_best_model.pth'
    polyA_length = 6481
    input_length = 12048
    target = Target.NONCODING

    # Set up
    logger = setup_logging()
    client = Client(logger)
    config = get_config(config_file)
    model = Model(model_file, config, logger)
    processor = SignalProcessor(polyA_length, input_length)
    riser = Riser(client, model, processor, logger)

    # Log initial setup
    # logger.info(" ".join(sys.argv)) # TODO: Replace below with this
    logger.info('Config file: %s', config_file)
    logger.info('Model file: %s', model_file)
    logger.info('PolyA + seq adapter length: %s', polyA_length)
    logger.info('Input length: %s', input_length)
    logger.info('Target: %s', target)

    # Run analysis
    client.start_streaming_reads()
    riser.enrich_sequencing_run(target)

    # Close read stream
    client.reset()
    logger.info('Client reset and live read stream ended.')


if __name__ == "__main__":
    main()
