import os
import pytest
import pandas as pd

from src.enc4ppm.frequency_encoder import FrequencyEncoder
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
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Ship',
        },
        {
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Receive Payment',
        },
        # Case002
        {
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Contact Supplier',
        },
        {
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 1,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Ship',
        },
        {
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 0,
            'Contact Supplier': 1,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Receive Payment',
        },
        # Case003
        {
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Ship',
        },
        {
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Receive Payment',
        },
        {
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 1,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Order Returned',
        },
        {
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 1,
            'Contact Supplier': 0,
            'Order Returned': 1,
            'Issue Refund': 0,
            'label': 'Issue Refund',
        },
    ]

def test_frequency_encoder(log, gt_encoded_log):
    frequency_encoder = FrequencyEncoder(
        case_id_key=CASE_ID_KEY,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
    )
    encoded_log = frequency_encoder.encode(log, labeling_type=LabelingType.NEXT_ACTIVITY)

    assert len(gt_encoded_log) == len(encoded_log)
    assert len(encoded_log.columns) == NUM_ACTIVITIES + 1 # +1 is label

    for row in gt_encoded_log:
        assert row in encoded_log.to_dict(orient='records')