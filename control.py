import time

import torch

from utilities import Species


class SequencerControl():
    def __init__(self, client, model, processor, logger, out_file):
        self.client = client
        self.model = model
        self.processor = processor
        self.logger = logger
        self.out_file = out_file

    def enrich(self, target, duration=0.1, interval=4.0):
        self.client.send_warning(
            'The sequencing run is being controlled by RISER, reads that are '
            'not in the target class will be ejected from the pore.')

        with open(f'{self.out_file}.csv', 'a') as out_file:
            self._write_header(out_file)

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
                    pred, probs = self._classify_signal(signal)
                    if self._reject(pred, target):
                        reads_to_reject.append((channel, read.number))
                    reads_processed.append((channel, read.number))
                    self._write(out_file, channel, read.id, probs, pred, target)

                # Send reject requests
                self.client.reject_reads(reads_to_reject, duration)
                self.client.finish_processing_reads(reads_processed)

                # Get ready for the next batch
                end = time.time()
                self._rest(start, end, interval)
                self.logger.info('Time to process batch of %d reads (%d rejected): %fs',
                    len(reads_processed),
                    len(reads_to_reject),
                    end - start)
            else:
                self.client.send_warning('RISER has stopped running.')
                self.logger.info('Client stopped.')

    def finish(self):
        self.client.reset()
        self.logger.info('Client reset and live read stream ended.')

    def _rest(self, start, end, interval):
        if start + interval > end:
            time.sleep(interval + start - end)

    def _classify_signal(self, signal):
        signal = self.processor.process(signal)
        probs = self.model.classify(signal)
        prediction = Species(torch.argmax(probs, dim=1).item())
        return prediction, probs
    
    def _reject(self, prediction, target):
        return prediction != target

    def _write_header(self, csv_file):
        csv_file.write('channel,read_id,probability_noncoding,'
                       'probability_coding,prediction,target,decision\n')

    def _write(self, csv_file, channel, read, probs, prediction, target):
        noncod_prob = probs[0][0]
        coding_prob = probs[0][1]
        decision = 'REJECT' if self._reject(prediction, target) else 'ACCEPT'
        csv_file.write(f'{read},{channel},{noncod_prob:.2f},{coding_prob:.2f},'
                       f'{prediction.name},{target.name},{decision}\n')
