import os
import pytest
import pandas as pd

from src.enc4ppm.complex_index_encoder import ComplexIndexEncoder
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
            CASE_ID_KEY: 'Case001',
            'Customer': 'CustomerA',
            'event_1': 'Receive Order',
            'Amount_1': 0,
            'event_2': 'PADDING',
            'Amount_2': 0,
            'event_3': 'PADDING',
            'Amount_3': 0,
            'event_4': 'PADDING',
            'Amount_4': 0,
            'event_5': 'PADDING',
            'Amount_5': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case001',
            'Customer': 'CustomerA',
            'event_1': 'Receive Order',
            'Amount_1': 0,
            'event_2': 'Ship',
            'Amount_2': 0,
            'event_3': 'PADDING',
            'Amount_3': 0,
            'event_4': 'PADDING',
            'Amount_4': 0,
            'event_5': 'PADDING',
            'Amount_5': 0,
            'label': 'Receive Payment',
        },
        # Case002
        {
            CASE_ID_KEY: 'Case002',
            'Customer': 'CustomerB',
            'event_1': 'Receive Order',
            'Amount_1': 0,
            'event_2': 'PADDING',
            'Amount_2': 0,
            'event_3': 'PADDING',
            'Amount_3': 0,
            'event_4': 'PADDING',
            'Amount_4': 0,
            'event_5': 'PADDING',
            'Amount_5': 0,
            'label': 'Contact Supplier',
        },
        {
            CASE_ID_KEY: 'Case002',
            'Customer': 'CustomerB',
            'event_1': 'Receive Order',
            'Amount_1': 0,
            'event_2': 'Contact Supplier',
            'Amount_2': -20,
            'event_3': 'PADDING',
            'Amount_3': 0,
            'event_4': 'PADDING',
            'Amount_4': 0,
            'event_5': 'PADDING',
            'Amount_5': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case002',
            'Customer': 'CustomerB',
            'event_1': 'Receive Order',
            'Amount_1': 0,
            'event_2': 'Contact Supplier',
            'Amount_2': -20,
            'event_3': 'Ship',
            'Amount_3': 0,
            'event_4': 'PADDING',
            'Amount_4': 0,
            'event_5': 'PADDING',
            'Amount_5': 0,
            'label': 'Receive Payment',
        },
        # Case003
        {
            CASE_ID_KEY: 'Case003',
            'Customer': 'CustomerA',
            'event_1': 'Receive Order',
            'Amount_1': 0,
            'event_2': 'PADDING',
            'Amount_2': 0,
            'event_3': 'PADDING',
            'Amount_3': 0,
            'event_4': 'PADDING',
            'Amount_4': 0,
            'event_5': 'PADDING',
            'Amount_5': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case003',
            'Customer': 'CustomerA',
            'event_1': 'Receive Order',
            'Amount_1': 0,
            'event_2': 'Ship',
            'Amount_2': 0,
            'event_3': 'PADDING',
            'Amount_3': 0,
            'event_4': 'PADDING',
            'Amount_4': 0,
            'event_5': 'PADDING',
            'Amount_5': 0,
            'label': 'Receive Payment',
        },
        {
            CASE_ID_KEY: 'Case003',
            'Customer': 'CustomerA',
            'event_1': 'Receive Order',
            'Amount_1': 0,
            'event_2': 'Ship',
            'Amount_2': 0,
            'event_3': 'Receive Payment',
            'Amount_3': 300,
            'event_4': 'PADDING',
            'Amount_4': 0,
            'event_5': 'PADDING',
            'Amount_5': 0,
            'label': 'Order Returned',
        },
        {
            CASE_ID_KEY: 'Case003',
            'Customer': 'CustomerA',
            'event_1': 'Receive Order',
            'Amount_1': 0,
            'event_2': 'Ship',
            'Amount_2': 0,
            'event_3': 'Receive Payment',
            'Amount_3': 300,
            'event_4': 'Order Returned',
            'Amount_4': 0,
            'event_5': 'PADDING',
            'Amount_5': 0,
            'label': 'Issue Refund',
        },
        # Case004
        {
            CASE_ID_KEY: 'Case004',
            'Customer': 'CustomerC',
            'event_1': 'Receive Order',
            'Amount_1': 0,
            'event_2': 'PADDING',
            'Amount_2': 0,
            'event_3': 'PADDING',
            'Amount_3': 0,
            'event_4': 'PADDING',
            'Amount_4': 0,
            'event_5': 'PADDING',
            'Amount_5': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case004',
            'Customer': 'CustomerC',
            'event_1': 'Receive Order',
            'Amount_1': 0,
            'event_2': 'Ship',
            'Amount_2': 0,
            'event_3': 'PADDING',
            'Amount_3': 0,
            'event_4': 'PADDING',
            'Amount_4': 0,
            'event_5': 'PADDING',
            'Amount_5': 0,
            'label': 'Receive Payment',
        },
    ]


def test_complex_index_encoder(log, gt_encoded_log):
    complex_index_encoder = ComplexIndexEncoder(
        labeling_type=LabelingType.NEXT_ACTIVITY,
        timestamp_format=TIMESTAMP_FORMAT,
        case_id_key=CASE_ID_KEY,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
    )
    encoded_log = complex_index_encoder.encode(
        log,
        static_attributes=['Customer'],
        dynamic_attributes=['Amount'],
    )

    assert len(gt_encoded_log) == len(encoded_log)
    assert len(encoded_log.columns) == 5*2 + 1 + 1 + 1 # 5*2 is max trace length (activity+amount), + 1 is case id, + 1 is customer (static attribute), + 1 is label

    encoded_log = encoded_log.to_dict(orient='records')
    for i in range(len(gt_encoded_log)):
        assert gt_encoded_log[i] == encoded_log[i]