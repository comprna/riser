from datetime import datetime
from enum import Enum

from attrdict import AttrDict
import yaml


DT_FORMAT = '%Y-%m-%dT%H:%M:%S'


class Species(Enum):
    NONCODING = 0
    CODING = 1

# TODO: Verify config
def get_config(filepath):
    with open(filepath) as config_file:
        return AttrDict(yaml.load(config_file, Loader=yaml.Loader))

def get_datetime_now():
    return datetime.now().strftime(DT_FORMAT)

