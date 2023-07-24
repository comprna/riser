import time

class SequencerControl():
    def __init__(self, client, model, processor, logger, out_file):
        self.client = client
        self.model = model
        self.proc = processor
        self.logger = logger
        self.out_filename = out_file

    # IMPORTANT: The target function in the split_flowcell branch ignores the
    # mode setting.  Instead, it splits the flowcell channels into thirds to
    # allow all conditions to be tested on a single flowcell, removing flowcell
    # variability as an experimental factor.
    #
    # Each consecutive channel alternates between control, enrich and deplete
    # so that the conditions are evenly distributed throughout the channels in
    # the flowcell.
    def target(self, mode, duration_h, threshold, unblock_duration=0.1):
        self.client.send_warning(
            'The sequencing run is being controlled by RISER using a split '
            'flowcell to test all conditions simultaneously.')
 
        with open(f'{self.out_filename}.csv', 'a') as out_file:
            self._write_header(out_file)
            run_start = time.monotonic()
            progress_time = run_start + 60
            duration_s = self._hours_to_seconds(duration_h)
            n_assessed = 0
            n_rejected = 0
            n_accepted = 0
            while self.client.is_running() and time.monotonic() < run_start + duration_s:
                # Get batch of reads to process
                batch_start = time.monotonic()
                reads_to_reject = []
                reads_to_accept = []
                reads_unclassified = []
                for channel, read in self.client.get_read_batch():
                    # Only process signal if it's long enough
                    signal = self.client.get_raw_signal(read)
                    if len(signal) < self.proc.get_min_assessable_length():
                        continue

                    # Classify the RNA class to which the read belongs
                    signal, max_length = self.proc.preprocess(signal)
                    p_off_target, p_on_target = self.model.classify(signal)
                    n_assessed += 1

                    # Determine mode based on channel number. Since channels are
                    # numbered 1-512 (inclusive), the following logic will
                    # assign 170 channels to enrich, 171 to deplete and 171 to
                    # control.  Reads from the 171st control channel and 171st
                    # deplete channel can be discarded later so that there are
                    # an equal number of channels per condition.
                    if channel % 3 == 0:
                        mode = "enrich"
                    elif channel % 3 == 1:
                        continue # Control
                    elif channel % 3 == 2:
                        mode = "deplete"

                    # Decide what to do with this read
                    if mode == "enrich" and p_on_target > threshold:
                        decision = "accept"
                    elif mode == "enrich" and p_off_target > threshold:
                        decision = "reject"
                    elif mode == "deplete" and p_on_target > threshold:
                        decision = "reject"
                    elif mode == "deplete" and p_off_target > threshold:
                        decision = "accept"
                    elif max_length == True and p_on_target < threshold \
                        and p_off_target < threshold:
                        decision = "no_decision"
                    else:
                        decision = "try_again"

                    # Process the read decision
                    if decision == "accept":
                        reads_to_accept.append((channel, read.number))
                    elif decision == "reject":
                        reads_to_reject.append((channel, read.number))
                    elif decision == "no_decision":
                        reads_unclassified.append((channel, read.number))
                    self._write(out_file, batch_start, channel, read.id, len(signal),
                                p_on_target, threshold, mode, decision)

                # Send reject requests
                self.client.reject_reads(reads_to_reject, unblock_duration)
                n_rejected += len(reads_to_reject)

                # Don't need to reassess the reads that were rejected, accepted
                # or couldn't be classified after the maximum input length
                done = reads_to_reject + reads_to_accept + reads_unclassified
                self.client.finish_processing_reads(done)
                n_accepted += len(reads_to_accept)

                # Log progress each minute
                if batch_start > progress_time:
                    self.logger.info(f"In the last minute {n_assessed} reads "
                                     f"were assessed, {n_accepted} were "
                                     f"accepted and {n_rejected} were rejected")
                    n_assessed = 0
                    n_rejected = 0
                    n_accepted = 0
                    progress_time = batch_start + 60
            else:
                self.client.send_warning('RISER has stopped running.')
                if not self.client.is_running():
                    self.logger.info('Client has stopped.')
                if time.monotonic() > run_start + duration_s:
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

    def _write_header(self, csv_file):
        csv_file.write('batch_start,read_id,channel,sig_length,prob_target,threshold,objective,decision\n')

    def _write(self, csv_file, batch_start, channel, read, sig_length,
               p_on_target, threshold, objective, decision):
        csv_file.write(f'{batch_start:.0f},{read},{channel},{sig_length},{p_on_target:.2f},{threshold},'
                       f'{objective},{decision}\n')
