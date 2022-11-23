from time import time, sleep

import torch

from utilities import Species


class SequencerControl():
    def __init__(self, client, model, processor, logger, out_file):
        self.client = client
        self.model = model
        self.processor = processor
        self.logger = logger
        self.out_filename = out_file

    def enrich(self, target, duration_h, unblock_duration=0.1, interval=1.0):
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
                for i, (channel, read) in enumerate(self.client.get_read_batch(),
                                                    start=1):
                    # Only process read if it's long enough
                    signal = self.client.get_raw_signal(read)
                    if len(signal) < self.processor.get_min_length(): continue

                    # Classify the RNA species to which the read belongs
                    pred, probs = self._classify_signal(signal)
                    if self._should_reject(pred, target):
                        reads_to_reject.append((channel, read.number))
                    reads_processed.append((channel, read.number))
                    self._write(out_file, channel, read.id, probs, pred, target)

                # Send reject requests
                self.client.reject_reads(reads_to_reject, unblock_duration)
                self.client.finish_processing_reads(reads_processed)

                # Get ready for the next batch
                batch_end = time()
                self._rest(batch_start, batch_end, interval)
                self.logger.info('Batch of %3d reads received: %2d long enough '
                                 'to assess, %2d of which were rejected (took '
                                 '%.4fs)',
                                 i,
                                 len(reads_processed),
                                 len(reads_to_reject),
                                 batch_end - batch_start)
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

    def _rest(self, start, end, interval):
        if start + interval > end:
            sleep(interval + start - end)

    def _classify_signal(self, signal):
        signal = self.processor.process(signal)
        probs = self.model.classify(signal)
        prediction = Species(torch.argmax(probs, dim=1).item())
        return prediction, probs
    
    def _should_reject(self, prediction, target):
        return prediction != target

    def _write_header(self, csv_file):
        csv_file.write('read_id,channel,probability_noncoding,'
                       'probability_coding,prediction,target,decision\n')

    def _write(self, csv_file, channel, read, probs, prediction, target):
        noncod_prob = probs[0][0]
        coding_prob = probs[0][1]
        decision = 'REJECT' if self._should_reject(prediction, target) else 'ACCEPT'
        csv_file.write(f'{read},{channel},{noncod_prob:.2f},{coding_prob:.2f},'
                       f'{prediction.name},{target.name},{decision}\n')
