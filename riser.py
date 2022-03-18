from datetime import datetime
import time


DT_FORMAT = '%Y-%m-%dT%H:%M:%S'


# TODO: Move inside RISER, private class method
def _get_datetime_now():
    return datetime.now().strftime(DT_FORMAT)


class Riser():
    def __init__(self, client, model, processor, logger):
        self.client = client
        self.model = model
        self.processor = processor
        self.logger = logger
        self.out_file = f'riser_{_get_datetime_now()}.csv'

    def enrich_sequencing_run(self, target, duration=0.1, throttle=4.0):
        self.client.send_warning(
            'The sequencing run is being controlled by RISER, reads that are '
            'not in the target class will be ejected from the pore.')

        with open(self.out_file, 'a') as f: # TODO: Refactor, nested code ugly

            while self.client.is_running():

                # Get batch of reads to process
                start_t = time.time()
                reads_processed = []
                reads_to_reject = []
                for (channel, read) in self.client.get_read_batch():
                    # Only process read if it's long enough
                    signal = self.client.get_raw_signal(read)
                    if len(signal) < self.processor.get_required_length(): # TODO: Rename get_min_length
                        continue

                    prediction = self.classify_signal(signal)
                    if prediction != target.value:
                       reads_to_reject.append((channel, read.number))

                    # Don't need to assess the same read twice
                    reads_processed.append((channel, read.number))

                    f.write(f'{channel},{read.number}')


                # Send reject requests
                self.client.reject_reads(reads_to_reject, duration)
                self.client.track_reads_processed(reads_processed)

                # Limit request rate
                end_t = time.time()
                if start_t + throttle > end_t:
                    time.sleep(throttle + start_t - end_t)
                self.logger.info('Time to process batch of %d reads (%d rejected): %fs',
                            len(reads_processed),
                            len(reads_to_reject),
                            end_t - start_t)
            else:
                self.client.send_warning('RISER has stopped running.')
                self.logger.info('Client stopped.')
    
    def classify_signal(self, signal):
        signal = self.processor.process(signal)
        return self.model.classify(signal) # TODO: Return prediction as enum value
