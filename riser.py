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

POLYA_LENGTH = 6481 # TODO: Remove globals
INPUT_LENGTH = 12048 # TODO: Remove globals


def analysis(client, model, device, processor, duration=0.1, throttle=0.4, batch_size=512):
    n_rejected = 0

    while client.is_running:

        # Initialise current batch of reads
        t0 = timer()
        unblock_batch_reads = []
        stop_receiving_reads = []

        # Iterate through reads in current batch
        for (_, read) in client.get_read_chunks(batch_size=batch_size, last=True):

            # Get raw signal
            raw_signal = np.frombuffer(read.raw_data, client.signal_dtype)

            # Classify signal if it is long enough
            if len(raw_signal) >= POLYA_LENGTH + INPUT_LENGTH:
                input_signal = processor.preprocess(raw_signal)
                with torch.no_grad():
                    input_signal = torch.from_numpy(input_signal)
                    input_signal = input_signal.unsqueeze(0) # Create mini-batch as expected by model
                    input_signal = input_signal.to(device, dtype=torch.float)
                    y = model(input_signal)
                    print(y)

        # Send reject requests
        if len(unblock_batch_reads) > 0:
            client.unblock_read_batch(unblock_batch_reads, duration=duration)
            client.stop_receiving_batch(stop_receiving_reads)

        # Count number rejected
        n_rejected += len(unblock_batch_reads)
        print(f"Total n reads rejected: {n_rejected}")

        # Limit request rate
        t1 = timer()
        if t0 + throttle > t1:
            time.sleep(throttle + t0 - t1)
        print(f"Time to unblock batch of {len(unblock_batch_reads):3} reads: {t1 - t0:.4f}s")
    else:
        print("Client stopped, finished analysis.")


def setup_client():
    read_until_client = ReadUntilClient(filter_strands=True,
                                        one_chunk=False,
                                        cache_type=AccumulatingCache)

    # Start communication with MinKNOW
    read_until_client.run(first_channel=1, last_channel=512)

    # Make sure client is running
    while read_until_client.is_running is False:
        time.sleep(0.1)


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


def main():
    # CL args
    config_file = './local_data/configs/train-cnn-20.yaml'
    model_file = 'local_data/models/train-cnn-20_0_best_model.pth'

    # Set up ReadUntil client
    client = setup_client()

    # Set up device to use for running model
    device = setup_device()

    # Set up model
    model = setup_model(model_file, config_file, device)

    # TODO: Is ThreadPoolExecutor needed? Readfish just calls analysis
    # function directly.
    # with ThreadPoolExecutor() as executor:
    #     executor.submit(analysis, read_until_client)

    processor = SignalProcessor(POLYA_LENGTH, INPUT_LENGTH)

    analysis(client, model, device, processor)

    # TODO: Close connection to client.


if __name__ == "__main__":
    main()

