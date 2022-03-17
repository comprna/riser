from enum import Enum
from datetime import datetime
import logging
import time
from timeit import default_timer as timer

from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
import numpy as np
from read_until import ReadUntilClient
from read_until.read_cache import AccumulatingCache
import torch
from torchinfo import summary

from cnn import ConvNet
from utilities import get_config
from preprocess import SignalProcessor


# TODO: Logging
# TODO: Multithreading
# TODO: Annotate function signatures (arg types, return type)
# TODO: Comments


def classify(signal, device, model):
    with torch.no_grad():
        X = torch.from_numpy(signal).unsqueeze(0)
        X = X.to(device, dtype=torch.float)
        logits = model(X)
        return torch.argmax(logits, dim=1)


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


def send_message_to_minknow(client, severity, message):
    """
    severity: Severity enum (value sent to API)
    """
    client.connection.log.send_user_message(user_message=message,
                                            severity=severity.value)


def analysis(client, model, device, processor, target, logger, duration=0.1, throttle=4.0, batch_size=512):
    send_message_to_minknow(client,
                            Severity.WARNING,
                            f'RISER will reject reads that are not {target}. '
                            'This will affect the sequencing run.')
    while client.is_running:
        # Iterate through current batch of reads retrieved from client
        start_t = time.time()
        unblock_batch_reads = []
        stop_receiving_reads = []
        for (channel, read) in client.get_read_chunks(batch_size=batch_size,
                                                      last=True):
            # Preprocess raw signal if it's long enough
            signal = np.frombuffer(read.raw_data, client.signal_dtype)
            if len(signal) < processor.get_required_length():
                continue
            signal = processor.process(signal)

            # Accept or reject read
            prediction = classify(signal, device, model)
            if prediction != target:
                unblock_batch_reads.append((channel, read.number))

            # Don't need to assess the same read twice
            stop_receiving_reads.append((channel, read.number))

        # Send reject requests
        if len(unblock_batch_reads) > 0:
            client.unblock_read_batch(unblock_batch_reads, duration=duration)
        if len(stop_receiving_reads) > 0:
            client.stop_receiving_batch(stop_receiving_reads)

        # Limit request rate
        end_t = time.time()
        if start_t + throttle > end_t:
            time.sleep(throttle + start_t - end_t)
        logger.info('Time to process batch of %d reads (%d rejected): %fs',
                     len(stop_receiving_reads),
                     len(unblock_batch_reads),
                     end_t - start_t)
    else:
        send_message_to_minknow(client,
                                Severity.WARNING,
                                f'RISER has stopped running.')
        logger.info("ReadUntil client stopped.")


def setup_client(logger):
    client = ReadUntilClient(filter_strands=True, #TODO: Is this needed?
                             one_chunk=False,
                             cache_type=AccumulatingCache)
    client.run(first_channel=1, last_channel=512)
    while client.is_running is False:
        time.sleep(0.1)
        logger.info('Waiting for client to start streaming live reads.')
    logger.info('Client is running.')
    return client


def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")
    return device


def setup_model(model_file, config_file, device):
    config = get_config(config_file)
    model = ConvNet(config.cnn).to(device)
    model.load_state_dict(torch.load(model_file))
    summary(model)
    model.eval()
    return model


def setup_logging():
    dt_format = '%Y-%m-%dT%H:%M:%S'
    now = datetime.now().strftime(dt_format)
    logging.basicConfig(filename=f'riser_{now}.log',
                        level=logging.DEBUG,
                        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                        datefmt=dt_format)

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
    client = setup_client(logger)
    device = setup_device()
    model = setup_model(model_file, config_file, device)
    processor = SignalProcessor(polyA_length, input_length)
    target_class = target.value

    # Log initial setup
    # logger.info(" ".join(sys.argv)) # TODO: Replace below with this
    logger.info('Config file: %s', config_file)
    logger.info('Model file: %s', model_file)
    logger.info('PolyA + seq adapter length: %s', polyA_length)
    logger.info('Input length: %s', input_length)
    logger.info('Target: %s', target)
    logger.info('Target class: %s', target_class)

    # Run analysis
    # TODO: Is ThreadPoolExecutor needed? Readfish just calls analysis
    # function directly.
    # with ThreadPoolExecutor() as executor:
    #     executor.submit(analysis, read_until_client)
    analysis(client, model, device, processor, target_class, logger)

    # Close read stream
    client.reset()
    logger.info('Client reset and live read stream ended.')


if __name__ == "__main__":
    main()
