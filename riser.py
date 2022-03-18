from enum import Enum
from datetime import datetime
import logging
import time
from timeit import default_timer as timer

from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
import numpy as np
import torch

from client import Client
from model import Model
from utilities import get_config
from preprocess import SignalProcessor


# TODO: Annotate function signatures (arg types, return type)
# TODO: Comments
# TODO: Extract model class to encapsulate PyTorch code

DT_FORMAT = '%Y-%m-%dT%H:%M:%S'


class Target(Enum):
    NONCODING = 0
    CODING = 1


class Severity(Enum):
    """
    This matches the severity values expected for messages received by the 
    MinKNOW API.
    """
    TRACE = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


def analysis(client, model, processor, target, logger, duration=0.1, throttle=4.0):
    client.send_message_to_minknow(
        Severity.WARNING,
        ('RISER will accept reads that are %s and reject all others. This will '
        'affect the sequencing run.' % (target.name.lower())))

    out_file = f'riser_{get_datetime_now()}.csv'
    with open(out_file, 'a') as f: # TODO: Refactor, nested code ugly
        while client.is_running():
            # Iterate through current batch of reads retrieved from client
            start_t = time.time()
            assessed_reads = []
            reads_to_reject = []
            for (channel, read) in client.get_read_chunks():
                # Preprocess raw signal if it's long enough
                signal = client.get_raw_signal(read)
                if len(signal) < processor.get_required_length(): # TODO: Rename get_min_length
                    continue
                signal = processor.process(signal)

                # Accept or reject read
                prediction = model.classify(signal) # TODO: Return prediction as enum value
                if prediction != target.value:
                    reads_to_reject.append((channel, read.number))
                f.write(f'{channel},{read.number}')

                # Don't need to assess the same read twice
                assessed_reads.append((channel, read.number))

            # Send reject requests
            client.reject_reads(reads_to_reject, duration)
            client.track_assessed_reads(assessed_reads)

            # Limit request rate
            end_t = time.time()
            if start_t + throttle > end_t:
                time.sleep(throttle + start_t - end_t)
            logger.info('Time to process batch of %d reads (%d rejected): %fs',
                        len(assessed_reads),
                        len(reads_to_reject),
                        end_t - start_t)
        else:
            client.send_message_to_minknow(Severity.WARNING,
                                           f'RISER has stopped running.')
            logger.info("ReadUntil client stopped.")



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
    
    # TODO: Riser class??
    # Handles setting up logger, client, device, model
    # Give it a processor

    # Set up
    logger = setup_logging()
    client = Client(logger)
    config = get_config(config_file)
    model = Model(model_file, config, logger)
    processor = SignalProcessor(polyA_length, input_length)

    # Log initial setup
    # logger.info(" ".join(sys.argv)) # TODO: Replace below with this
    logger.info('Config file: %s', config_file)
    logger.info('Model file: %s', model_file)
    logger.info('PolyA + seq adapter length: %s', polyA_length)
    logger.info('Input length: %s', input_length)
    logger.info('Target: %s', target)

    # Run analysis
    # TODO: Is ThreadPoolExecutor needed? Readfish just calls analysis
    # function directly.
    # with ThreadPoolExecutor() as executor:
    #     executor.submit(analysis, read_until_client)
    client.start_streaming_reads()
    analysis(client, model, processor, target, logger)

    # Close read stream
    client.reset()
    logger.info('Client reset and live read stream ended.')


if __name__ == "__main__":
    main()
