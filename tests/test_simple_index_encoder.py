import os
import pytest
import pandas as pd

from src.enc4ppm.simple_index_encoder import SimpleIndexEncoder
from src.enc4ppm.constants import LabelingType, CategoricalEncoding
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
            'event_1': 'Receive Order',
            'event_2': 'PADDING',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case001',
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Receive Payment',
        },
        # Case002
        {
            CASE_ID_KEY: 'Case002',
            'event_1': 'Receive Order',
            'event_2': 'PADDING',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Contact Supplier',
        },
        {
            CASE_ID_KEY: 'Case002',
            'event_1': 'Receive Order',
            'event_2': 'Contact Supplier',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case002',
            'event_1': 'Receive Order',
            'event_2': 'Contact Supplier',
            'event_3': 'Ship',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Receive Payment',
        },
        # Case003
        {
            CASE_ID_KEY: 'Case003',
            'event_1': 'Receive Order',
            'event_2': 'PADDING',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case003',
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Receive Payment',
        },
        {
            CASE_ID_KEY: 'Case003',
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'Receive Payment',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'label': 'Order Returned',
        },
        {
            CASE_ID_KEY: 'Case003',
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'Receive Payment',
            'event_4': 'Order Returned',
            'event_5': 'PADDING',
            'label': 'Issue Refund',
        },
    ]

@pytest.fixture
def gt_encoded_log_latest_payload():
    return [
        # Case001
        {
            CASE_ID_KEY: 'Case001',
            'event_1': 'Receive Order',
            'event_2': 'PADDING',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'Customer_latest': 'CustomerA',
            'Amount_latest': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case001',
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'Customer_latest': 'CustomerA',
            'Amount_latest': 0,
            'label': 'Receive Payment',
        },
        # Case002
        {
            CASE_ID_KEY: 'Case002',
            'event_1': 'Receive Order',
            'event_2': 'PADDING',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'Customer_latest': 'CustomerB',
            'Amount_latest': 0,
            'label': 'Contact Supplier',
        },
        {
            CASE_ID_KEY: 'Case002',
            'event_1': 'Receive Order',
            'event_2': 'Contact Supplier',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'Customer_latest': 'CustomerB',
            'Amount_latest': -20,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case002',
            'event_1': 'Receive Order',
            'event_2': 'Contact Supplier',
            'event_3': 'Ship',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'Customer_latest': 'CustomerB',
            'Amount_latest': 0,
            'label': 'Receive Payment',
        },
        # Case003
        {
            CASE_ID_KEY: 'Case003',
            'event_1': 'Receive Order',
            'event_2': 'PADDING',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'Customer_latest': 'CustomerA',
            'Amount_latest': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case003',
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'PADDING',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'Customer_latest': 'CustomerA',
            'Amount_latest': 0,
            'label': 'Receive Payment',
        },
        {
            CASE_ID_KEY: 'Case003',
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'Receive Payment',
            'event_4': 'PADDING',
            'event_5': 'PADDING',
            'Customer_latest': 'CustomerA',
            'Amount_latest': 300,
            'label': 'Order Returned',
        },
        {
            CASE_ID_KEY: 'Case003',
            'event_1': 'Receive Order',
            'event_2': 'Ship',
            'event_3': 'Receive Payment',
            'event_4': 'Order Returned',
            'event_5': 'PADDING',
            'Customer_latest': 'CustomerA',
            'Amount_latest': 0,
            'label': 'Issue Refund',
        },
    ]

def test_simple_index_encoder(log, gt_encoded_log):
    simple_index_encoder = SimpleIndexEncoder(
        labeling_type=LabelingType.NEXT_ACTIVITY,
        case_id_key=CASE_ID_KEY,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
    )
    encoded_log = simple_index_encoder.encode(log)

    assert len(gt_encoded_log) == len(encoded_log)
    assert len(encoded_log.columns) == 5 + 1 + 1 # 5 is max trace length, + 1 is case id, + 1 is label

    print(f'encoded log')
    print(encoded_log.to_dict(orient='records'))
    for row in gt_encoded_log:
        print(f'Asserting row = {row}')
        assert row in encoded_log.to_dict(orient='records')


def test_simple_index_encoder_latest_payload(log, gt_encoded_log_latest_payload):
    simple_index_encoder = SimpleIndexEncoder(
        labeling_type=LabelingType.NEXT_ACTIVITY,
        case_id_key=CASE_ID_KEY,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
    )
    encoded_log = simple_index_encoder.encode(
        log,
        include_latest_payload=True,
        attributes=['Customer', 'Amount'],
        categorical_attributes_encoding=CategoricalEncoding.STRING,
    )

    assert len(gt_encoded_log_latest_payload) == len(encoded_log)
    assert len(encoded_log.columns) == 5 + 1 + 1 + 2 # 5 is max trace length, + 1 is case id, + 1 is label, + 2 are attributes

    for row in gt_encoded_log_latest_payload:
        assert row in encoded_log.to_dict(orient='records')