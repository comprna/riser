from time import time, sleep

import torch


class SequencerControl():
    def __init__(self, client, model, processor, logger, out_file):
        self.client = client
        self.model = model
        self.proc = processor
        self.logger = logger
        self.out_filename = out_file

    # TODO: Expand to enable enrichment or depletion
    def enrich(self, duration_h, threshold, unblock_duration=0.1):
        self.client.send_warning(
            'The sequencing run is being controlled by RISER, reads that are '
            'not in the target class will be ejected from the pore.')
 
        with open(f'{self.out_filename}.csv', 'a') as out_file:
            self._write_header(out_file)

            run_start = time()
            duration_s = self._hours_to_seconds(duration_h)
            while self.client.is_running() and time() < run_start + duration_s:
                # Get batch of reads to process
                batch_start = time()
                reads_processed = []
                reads_to_reject = []
                i = 0
                n_assessed = 0
                for i, (channel, read) in enumerate(self.client.get_read_batch(),
                                                    start=1):
                    # Only process signal if it's within the assessable length 
                    # range
                    signal = self.client.get_raw_signal(read)
                    if len(signal) < self.proc.get_min_assessable_length() or \
                        len(signal) > self.proc.get_max_assessable_length():
                        continue

                    # Classify the RNA class to which the read belongs
                    p_off_target, p_on_target = self._classify_signal(signal)
                    n_assessed += 1

                    # To enrich the target class, the off-target class gets 
                    # rejected (if the probability of the off-target class 
                    # exceeds the classifier threshold).
                    # If the classifier threshold is not exceeded, a confident
                    # prediction cannot be made and so we do nothing.
                    if p_off_target < threshold:
                        self._write(out_file, channel, read.id, p_on_target,
                                    threshold, "enrich", "no_decision")
                        continue

                    # Reject
                    reads_to_reject.append((channel, read.number))
                    self._write(out_file, channel, read.id, p_on_target,
                                threshold, "enrich", "reject")

                # Send reject requests
                self.client.reject_reads(reads_to_reject, unblock_duration)
                self.client.finish_processing_reads(reads_to_reject)

                # Get ready for the next batch
                batch_end = time()
                self.logger.info(f"Batch of {i} reads received: {n_assessed} "
                                 f"in assessable length range, "
                                 f"{len(reads_to_reject)} of which were "
                                 f"rejected (took {batch_end-batch_start:.4f}s)")
            else:
                self.client.send_warning('RISER has stopped running.')
                if not self.client.is_running():
                    self.logger.info('Client has stopped.')
                if time () > run_start + duration_s:
                    self.logger.info(f'RISER has timed out after {duration_h} '
                                     'hours as requested.')

    def start(self):
        self.client.start_streaming_reads()
        self.logger.info('Live read stream started.')

    def finish(self):
        self.client.reset()
        self.logger.info('Client reset and live read stream ended.')

    def _hours_to_seconds(self, hours):
        return hours * 60 * 60

    def _classify_signal(self, signal):
        signal = self.proc.trim_polyA(signal)
        signal = self.proc.mad_normalise(signal)
        probs = self.model.classify(signal)
        return probs

    def _write_header(self, csv_file):
        csv_file.write('read_id,channel,prob_target,threshold,objective,decision\n')

    def _write(self, csv_file, channel, read, p_on_target, threshold, objective, decision):
        csv_file.write(f'{read},{channel},{p_on_target:.2f},{threshold},'
                       f'{objective},{decision}\n')
