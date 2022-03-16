import time
from timeit import default_timer as timer

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from read_until import ReadUntilClient
from read_until.read_cache import AccumulatingCache
import torch
from torchinfo import summary

from cnn import ConvNet
from utilities import get_config

POLYA_SIGNAL = 6481
MODEL_INPUT = 12048

def analysis(client, model, device, duration=0.1, throttle=0.4, batch_size=512):
    n_rejected = 0
    t_start = time.time()

    # Perform analysis for 30 mins
    while client.is_running and time.time() < t_start + 1800: # TODO: Remove

        # Initialise current batch of reads to reject
        t0 = timer()
        i = 0
        unblock_batch_reads = []
        stop_receiving_reads = []

        # Iterate through reads in current batch
        for i, (channel, read) in enumerate(
            client.get_read_chunks(batch_size=batch_size, last=True),
            start=1):

            # Get raw signal
            raw_data = np.frombuffer(read.raw_data, client.signal_dtype)

            # Classify signal if it is long enough
            if len(raw_data) >= POLYA_SIGNAL + MODEL_INPUT:
                X = raw_data[:MODEL_INPUT]
                with torch.no_grad():
                    X = torch.from_numpy(X)
                    X = X.unsqueeze(0) # Create mini-batch as expected by model
                    X = X.to(device, dtype=torch.float)
                    y = model(X)
                    print(y)

            # Mark reads to be rejected only if they are long enough
            if len(raw_data) >= POLYA_SIGNAL + MODEL_INPUT:
                print(f'{len(raw_data)} LONG ENOUGH******************')
                unblock_batch_reads.append((channel, read.number))
                stop_receiving_reads.append((channel, read.number))
            else:
                print(f'{len(raw_data)} TOO SHORT')

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


def main():
    # Set up ReadUntil client
    read_until_client = ReadUntilClient(filter_strands=True,
                                        one_chunk=False,
                                        cache_type=AccumulatingCache)

    # Start client communication with MinKNOW
    read_until_client.run(first_channel=1, last_channel=512)

    # Make sure client is running before starting analysis
    while read_until_client.is_running is False:
        time.sleep(0.1)

    # Set up device to use for running model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")

    # Load model config
    config_file = './local_data/configs/train-cnn-20.yaml'
    config = get_config(config_file)
    
    # Set up model
    model_file = 'local_data/models/train-cnn-20_0_best_model.pth'
    model = ConvNet(config.cnn).to(device)
    model.load_state_dict(torch.load(model_file))
    summary(model)
    model.eval()

    # TODO: Is ThreadPoolExecutor needed? Readfish just calls analysis
    # function directly.
    # with ThreadPoolExecutor() as executor:
    #     executor.submit(analysis, read_until_client)

    analysis(read_until_client, model, device)

    # TODO: Close connection to client.


if __name__ == "__main__":
    main()

