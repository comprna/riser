from asyncore import read
import time
from timeit import default_timer as timer

from concurrent.futures import ThreadPoolExecutor
from read_until import ReadUntilClient
from read_until.read_cache import AccumulatingCache



def analysis(client, duration=0.1, throttle=0.4, batch_size=512):
    while client.is_running:
        t0 = timer()
        i = 0
        unblock_batch_reads = []
        stop_receiving_reads = []
        for i, (channel, read) in enumerate(
            client.get_read_chunks(batch_size=batch_size, last=True),
            start=1):
            unblock_batch_reads.append((channel, read.number))
            stop_receiving_reads.append((channel, read.number))

        if len(unblock_batch_reads) > 0:
            client.unblock_read_batch(unblock_batch_reads, duration=duration)
            client.stop_receiving_batch(stop_receiving_reads)

        t1 = timer()
        # Limit request rate
        if t0 + throttle > t1:
            time.sleep(throttle + t0 - t1)
        print(f"Time to unblock batch of {i:3} reads: {t1 - t0:.4f}s")
    else:
        print("Client stopped, finished analysis.")


# Set one_chunk to false, otherwise client.get_read_chunks will only
# return very few reads - potentially caused by stop_receiving_read
# being called too frequently.
read_until_client = ReadUntilClient(filter_strands=True,
                                    one_chunk=False)

read_until_client.run(first_channel=1, last_channel=512)

# Make sure client is running before starting analysis
while read_until_client.is_running is False:
    time.sleep(0.1)

# TODO: Is ThreadPoolExecutor needed? Readfish just calls analysis
# function directly.
# with ThreadPoolExecutor() as executor:
#     executor.submit(analysis, read_until_client)

analysis(read_until_client)
