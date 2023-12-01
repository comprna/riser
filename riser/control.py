import time

class SequencerControl():
    def __init__(self, client, models, processor, logger, out_file):
        self.client = client
        self.models = models
        self.proc = processor
        self.logger = logger
        self.out_filename = out_file

    # IMPORTANT: The target function in the split_flowcell branch ignores the
    # mode setting.  Instead, it splits the flowcell channels into two groups to
    # allow both depletion and control conditions to be tested on a single
    # flowcell, removing flowcell variability as an experimental factor.
    #
    # Each consecutive channel alternates between control and deplete
    # so that the conditions are evenly distributed throughout the channels in
    # the flowcell.
    def target(self, mode, duration_h, threshold, unblock_duration=0.1):
        msg = "The sequencing run is being controlled by RISER using a split flowcell to test all conditions simultaneously."
        print(msg)
        self.client.send_warning(msg)
 
        with open(f'{self.out_filename}.csv', 'a') as out_file:
            self._write_header(out_file)
            run_start = time.monotonic()
            progress_time = run_start + 60
            duration_s = self._hours_to_seconds(duration_h)
            n_assessed = 0
            n_rejected = 0
            n_accepted = 0
            polyA_cache = {}
            while self.client.is_running() and time.monotonic() < run_start + duration_s:
                # Get batch of reads to process
                batch_start = time.monotonic()
                reads_to_reject = []
                reads_to_accept = []
                reads_unclassified = []
                for channel, read in self.client.get_read_batch():
                    # Determine mode based on channel number. Since channels are
                    # numbered 1-512 (inclusive), the following logic will
                    # assign 256 channels to deplete and 256 to control.
                    if channel % 2 == 0:
                        mode = "deplete"
                    else:
                        continue # Control

                    # Preprocess the signal
                    signal = self.client.get_raw_signal(read)

                    # Attempt to trim the polyA tail
                    signal, is_polyA_trimmed = self.proc.trim_polyA(signal, read.id, polyA_cache)

                    # If we haven't found the polyA yet
                    if not is_polyA_trimmed:

                        # Use a fixed trim length if enough time has passed
                        if self.proc.should_trim_fixed_length(signal):
                            signal = self.proc.trim_polyA_fixed_length(signal)

                            # Show max input length to network
                            signal = signal[:self.proc.get_max_length()]

                        # Otherwise, try again the next time we see this read
                        else:
                            continue

                    # If we have trimmed the polyA
                    else:
                        # Make sure signal is long enough to be assessed
                        if len(signal) < self.proc.get_min_length():
                            continue

                        # Trim signal if it is too long
                        if len(signal) > self.proc.get_max_length():
                            signal = signal[:self.proc.get_max_length()]

                    # Normalise
                    signal = self.proc.mad_normalise(signal)

                    # Classify the RNA class to which the read belongs
                    p_on_targets = []
                    p_off_targets = []
                    for model in self.models:
                        p_off_target, p_on_target = model.classify(signal)
                        p_off_targets.append(p_off_target)
                        p_on_targets.append(p_on_target)
                    n_assessed += 1

                    # Decide what to do with this read
                    if any(p > threshold for p in p_on_targets):
                        decision = "accept" if mode == "enrich" else "reject"
                    elif all(p > threshold for p in p_off_targets):
                        decision = "accept" if mode == "deplete" else "reject"
                    elif self.proc.is_max_length(signal):
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
                    self._write(out_file, batch_start, channel, read.id,
                                len(signal), self.models, p_on_targets,
                                threshold, mode, decision)

                    # Clear the polyA cache every 1000 reads
                    if len(polyA_cache) >= 1000:
                        polyA_cache = {}

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
                    self.logger.info(f"In the last minute {n_assessed} signals "
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
        csv_file.write('batch_start,read_id,channel,sig_length,models,prob_targets,threshold,mode,decision\n')

    def _write(self, csv_file, batch_start, channel, read, sig_length,
               models, p_on_targets, threshold, mode, decision):
        csv_file.write(f'{batch_start:.0f},{read},{channel},{sig_length},'
                       f'{";".join([m.target for m in models])},'
                       f'{";".join([str(p.item()) for p in p_on_targets])},'
                       f'{threshold},{mode},{decision}\n')
