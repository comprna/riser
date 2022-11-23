from enum import Enum
import time

import numpy as np

from read_until import ReadUntilClient
from read_until.read_cache import AccumulatingCache

"""
Advantages: https://softwareengineering.stackexchange.com/questions/298145/wrapping-third-party-library-is-best-practice#:~:text=In%20fact%2C%20wrapping%20third%2Dparty,are%20testing%20your%20own%20code.
"""

N_CHANNELS = 512


class Severity(Enum):
    """
    This matches the severity values expected for messages received by the 
    MinKNOW API.
    """
    TRACE = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class Client():
    def __init__(self, logger):
        self.logger = logger
        
        self.ru_client = ReadUntilClient(filter_strands=True,
                                         one_chunk=False,
                                         cache_type=AccumulatingCache)

    def start_streaming_reads(self):
        self.ru_client.run(first_channel=1, last_channel=N_CHANNELS)
        while self.ru_client.is_running is False:
            time.sleep(0.1)
            self.logger.info('Waiting for client to start streaming live reads.')
        self.logger.info('Client is running.')
    
    def is_running(self):
        return self.ru_client.is_running
    
    def get_read_batch(self):
        return self.ru_client.get_read_chunks(batch_size=N_CHANNELS, last=True)
    
    def get_raw_signal(self, read):
        return np.frombuffer(read.raw_data, self.ru_client.signal_dtype)
    
    def reject_reads(self, reads, unblock_duration):
        if reads:
            self.ru_client.unblock_read_batch(reads,
                                              duration=unblock_duration)

    def finish_processing_reads(self, reads):
        if reads:
            self.ru_client.stop_receiving_batch(reads)
    
    def reset(self):
        self.ru_client.reset()

    def send_warning(self, message):
        self._send_message(Severity.WARNING, message)

    def _send_message(self, severity, message):
        """
        severity: Severity enum (value sent to API)
        """
        self.ru_client.connection.log.send_user_message(user_message=message,
                                                        severity=severity.value)
                                
