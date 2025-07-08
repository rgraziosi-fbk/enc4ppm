from enum import Enum

class LabelingType(Enum):
    NEXT_ACTIVITY = 'next_activity'
    REMAINING_TIME = 'remaining_time'
    OUTCOME = 'outcome'
    CUSTOM = 'custom'

class CategoricalEncoding(Enum):
    STRING = 'string'
    ONE_HOT = 'one_hot'
