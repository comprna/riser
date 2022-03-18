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

    def enrich_sequencing_run(self, target, duration=0.1, throttle=4.0):
        self.client.send_warning(
            'The sequencing run is being controlled by RISER, reads that are '
            'not in the target class will be ejected from the pore.')

        out_file = f'riser_{_get_datetime_now()}.csv'
        with open(out_file, 'a') as f: # TODO: Refactor, nested code ugly
            while self.client.is_running():
                # Iterate through current batch of reads retrieved from client
                start_t = time.time()
                assessed_reads = []
                reads_to_reject = []
                for (channel, read) in self.client.get_read_chunks():
                    # Preprocess raw signal if it's long enough
                    signal = self.client.get_raw_signal(read)
                    if len(signal) < self.processor.get_required_length(): # TODO: Rename get_min_length
                        continue
                    signal = self.processor.process(signal)

                    # Accept or reject read
                    prediction = self.model.classify(signal) # TODO: Return prediction as enum value
                    if prediction != target.value:
                        reads_to_reject.append((channel, read.number))
                    f.write(f'{channel},{read.number}')

                    # Don't need to assess the same read twice
                    assessed_reads.append((channel, read.number))

                # Send reject requests
                self.client.reject_reads(reads_to_reject, duration)
                self.client.track_assessed_reads(assessed_reads)

                # Limit request rate
                end_t = time.time()
                if start_t + throttle > end_t:
                    time.sleep(throttle + start_t - end_t)
                self.logger.info('Time to process batch of %d reads (%d rejected): %fs',
                            len(assessed_reads),
                            len(reads_to_reject),
                            end_t - start_t)
            else:
                self.client.send_warning('RISER has stopped running.')
                self.logger.info('Client stopped.')
    
