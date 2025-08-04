import time
import cProfile
import pstats
import io
import pod5 


class SequencerControl:
    def __init__(self, models, processor, logger, out_file):
        # removed as to simulate online
        self.models = models
        self.proc = processor
        self.logger = logger
        self.out_filename = out_file

    def target(self, mode, threshold, pod5_file_path, limit=None):
        if limit is None:
            limit = 2000  # Default limit if not provided
        # No live warning or client checks
        with open(f"{self.out_filename}.csv", "a") as out_file, pod5.Reader(
            pod5_file_path
        ) as reader:
            self._write_header(out_file)
            n_assessed = 0
            n_rejected = 0
            n_accepted = 0
            polyA_cache = {}
            reads_to_reject = []  # For logging only (no actual rejection)
            reads_to_accept = []
            reads_unclassified = []

            # cProfile to see how long it takes
            pr = cProfile.Profile()
            pr.enable()

            # Iterate over all reads in POD5
            batch_start = time.monotonic()
            for read_record in reader.reads():
                if n_assessed >= limit:
                    break  # Stop after reaching the limit
                # Preprocess the signal for pod5
                signal = read_record.signal
                read_id = str(read_record.read_id)  # UUID strings to plain strings


                # Attempt to trim the polyA tail 
                signal, is_polyA_trimmed = self.proc.trim_polyA(
                    signal, read_id, polyA_cache
                )

                # If we haven't found the polyA yet
                if not is_polyA_trimmed:
                    
                    # Use a fixed trim length if enough time has passed
                    if self.proc.should_trim_fixed_length(signal):
                        signal = self.proc.trim_polyA_fixed_length(signal)
                        
                        # Show max input length to network
                        signal = signal[: self.proc.get_max_length()]
                        
                    # Otherwise, skip (no "try again" in offline mode)
                    else:
                        continue

                # If we have trimmed the polyA
                else:
                    
                    # Make sure signal is long enough to be assessed
                    if len(signal) < self.proc.get_min_length():
                        continue
                    
                    # Trim signal if it is too long
                    if len(signal) > self.proc.get_max_length():
                        signal = signal[: self.proc.get_max_length()]

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

                # Process the read decision for logging
                if decision == "accept":
                    reads_to_accept.append((channel, read_id))
                elif decision == "reject":
                    reads_to_reject.append((channel, read_id))
                elif decision == "no_decision":
                    reads_unclassified.append((channel, read_id))
                self._write(
                    out_file,
                    batch_start,
                    channel,
                    read_id,
                    len(signal),
                    self.models,
                    p_on_targets,
                    threshold,
                    mode,
                    decision,
                )

                # Clear the polyA cache every 1000 reads
                if len(polyA_cache) >= 1000:
                    polyA_cache = {}



            # End the profiling
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
            ps.print_stats()
            self.logger.info("cProfile Output:\n" + s.getvalue())  
            with open(f"{self.out_filename}_profile.txt", "w") as prof_file:
                prof_file.write(s.getvalue())  # Save to file

            n_rejected = len(reads_to_reject)
            n_accepted = len(reads_to_accept)
            self.logger.info(
                f"Total: {n_assessed} signals assessed, {n_accepted} accepted, {n_rejected} rejected"
            )

    def start(self):
        self.logger.info("POD5 processing started.")

    def finish(self):
        self.logger.info("POD5 processing ended.")

    def _hours_to_seconds(self, hours):
        return hours * 60 * 60

    def _get_read_id(self, read):
        return str(read.read_id)  

    def _write_header(self, csv_file):
        csv_file.write(
            "batch_start,read_id,channel,sig_length,models,prob_targets,threshold,mode,decision\n"
        )

    def _write(
        self,
        csv_file,
        batch_start,
        channel,
        read,
        sig_length,
        models,
        p_on_targets,
        threshold,
        mode,
        decision,
    ):
        csv_file.write(
            f"{batch_start:.0f},{read},{channel},{sig_length},"
            f'{";".join([m.target for m in models])},'
            f'{";".join([str(p.item()) for p in p_on_targets])},'
            f"{threshold},{mode},{decision}\n"
        )
