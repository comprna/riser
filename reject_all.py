import time
from timeit import default_timer as timer

from read_until import ReadUntilClient


def reject_all(client, duration=0.1, throttle=0.4, batch_size=512):
    # Reject reads as long as client is running
    while client.is_running:

        # Initialise current batch of reads to reject
        t0 = timer()
        i = 0
        unblock_batch_reads = []
        stop_receiving_reads = []

        # Iterate through reads in current batch
        for i, (channel, read) in enumerate(
            client.get_read_chunks(batch_size=batch_size, last=True),
            start=1):

            unblock_batch_reads.append((channel, read.number))
            stop_receiving_reads.append((channel, read.number))

        # Reject all reads
        if len(unblock_batch_reads) > 0:
            client.unblock_read_batch(unblock_batch_reads, duration=duration)
            client.stop_receiving_batch(stop_receiving_reads)

        # Limit request rate
        t1 = timer()
        if t0 + throttle > t1:
            time.sleep(throttle + t0 - t1)
        print(f"Time to unblock batch of {i:3} reads: {t1 - t0:.4f}s")
    else:
        print("Client stopped, finished analysis.")


def main():
    read_until_client = ReadUntilClient(filter_strands=True,
                                        one_chunk=False)

    read_until_client.run(first_channel=1, last_channel=512)

    # Make sure client is running before starting analysis
    while read_until_client.is_running is False:
        time.sleep(0.1)

    reject_all(read_until_client)


if __name__ == "__main__":
    main()
