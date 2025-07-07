import os
import pytest
import pandas as pd

from src.enc4ppm.simple_index_encoder import SimpleIndexEncoder
from src.enc4ppm.constants import LabelingType
from tests.data.test_log_info import *

@pytest.fixture
def log():
    log_path = os.path.join(os.path.dirname(__file__), 'data', TEST_LOG_NAME)
    return pd.read_csv(log_path)

@pytest.fixture
def gt_encoded_log():
    return [
        # Case001
        {
            'event_1': 'Receive Order',
            'event_2': 'PADDING',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Ship',
        },
        {
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Receive Payment',
        },
        # Case002
        {
            'event_1': 'Receive Order',
            'event_2': 'PADDING',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Contact Supplier',
        },
        {
            'event_1': 'Receive Order',
            'event_2': 'Contact Supplier',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Ship',
        },
        {
            'event_1': 'Receive Order',
            'event_2': 'Contact Supplier',
            'event_3': 'Ship',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Receive Payment',
        },
        # Case003
        {
            'event_1': 'Receive Order',
            'event_2': 'PADDING',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Ship',
        },
        {
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Receive Payment',
        },
        {
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'Receive Payment',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Order Returned',
        },
        {
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'Receive Payment',
            'event_4': 'Order Returned',
            'event_5': 'PADDING',
            'label': 'Issue Refund',
        },
    ]

def test_simple_index_encoder(log, gt_encoded_log):
    simple_index_encoder = SimpleIndexEncoder(
        case_id_key=CASE_ID_KEY,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
    )
    encoded_log = simple_index_encoder.encode(log, labeling_type=LabelingType.NEXT_ACTIVITY)

    assert len(gt_encoded_log) == len(encoded_log)
    assert len(encoded_log.columns) == 5 + 1 # 5 is max trace length, 1 is label

    print(f'encoded log')
    print(encoded_log.to_dict(orient='records'))
    for row in gt_encoded_log:
        print(f'Asserting row = {row}')
        assert row in encoded_log.to_dict(orient='records')