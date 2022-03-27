from datetime import datetime
import time

from utilities import get_datetime_now


class SequencerControl():
    def __init__(self, client, model, processor, logger):
        self.client = client
        self.model = model
        self.processor = processor
        self.logger = logger
        self.out_file = open(f'riser_{get_datetime_now()}.csv', 'a') # TODO: Move outside

    def enrich(self, target, duration=0.1, interval=4.0):
        self.client.send_warning(
            'The sequencing run is being controlled by RISER, reads that are '
            'not in the target class will be ejected from the pore.')

        while self.client.is_running():
            # Get batch of reads to process
            start = time.time()
            reads_processed = []
            reads_to_reject = []
            for (channel, read) in self.client.get_read_batch():

                # Only process read if it's long enough
                signal = self.client.get_raw_signal(read)
                if len(signal) < self.processor.get_min_length(): continue

                # Classify the RNA species to which the read belongs
                prediction = self._classify_signal(signal)
                if prediction != target:
                    reads_to_reject.append((channel, read.number))
                reads_processed.append((channel, read.number))
                self.out_file.write(f'{channel},{read.number}\n') # TODO: Do in batch?

            # Send reject requests
            self.client.reject_reads(reads_to_reject, duration)
            self.client.finish_processing_reads(reads_processed)

            # Get ready for the next batch
            end = time.time()
            self.rest(start, end, interval)
            self.logger.info('Time to process batch of %d reads (%d rejected): %fs',
                len(reads_processed),
                len(reads_to_reject),
                end - start)
        else:
            self.client.send_warning('RISER has stopped running.')
            self.logger.info('Client stopped.')
            self.out_file.close()

    def finish(self):
        if not self.out_file.closed:
            self.out_file.close()
        self.client.reset()

    def rest(self, start, end, interval):
        if start + interval > end:
            time.sleep(interval + start - end)
    
    def _classify_signal(self, signal):
        signal = self.processor.process(signal)
        return self.model.classify(signal) # TODO: Return prediction as enum value
