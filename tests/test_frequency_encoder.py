import os
import pytest
import pandas as pd

from src.enc4ppm.frequency_encoder import FrequencyEncoder
from src.enc4ppm.constants import LabelingType, CategoricalEncoding, PrefixStrategy
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
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case001',
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
            CASE_ID_KEY: 'Case002',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Contact Supplier',
        },
        {
            CASE_ID_KEY: 'Case002',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 1,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case002',
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
            CASE_ID_KEY: 'Case003',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case003',
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Receive Payment',
        },
        {
            CASE_ID_KEY: 'Case003',
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 1,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Order Returned',
        },
        {
            CASE_ID_KEY: 'Case003',
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 1,
            'Contact Supplier': 0,
            'Order Returned': 1,
            'Issue Refund': 0,
            'label': 'Issue Refund',
        },
    ]

@pytest.fixture
def gt_encoded_log_latest_payload():
    return [
        # Case001
        {
            CASE_ID_KEY: 'Case001',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'Customer_latest': 'CustomerA',
            'Amount_latest': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case001',
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'Customer_latest': 'CustomerA',
            'Amount_latest': 0,
            'label': 'Receive Payment',
        },
        # Case002
        {
            CASE_ID_KEY: 'Case002',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'Customer_latest': 'CustomerB',
            'Amount_latest': 0,
            'label': 'Contact Supplier',
        },
        {
            CASE_ID_KEY: 'Case002',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 1,
            'Order Returned': 0,
            'Issue Refund': 0,
            'Customer_latest': 'CustomerB',
            'Amount_latest': -20,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case002',
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 0,
            'Contact Supplier': 1,
            'Order Returned': 0,
            'Issue Refund': 0,
            'Customer_latest': 'CustomerB',
            'Amount_latest': 0,
            'label': 'Receive Payment',
        },
        # Case003
        {
            CASE_ID_KEY: 'Case003',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'Customer_latest': 'CustomerA',
            'Amount_latest': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case003',
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'Customer_latest': 'CustomerA',
            'Amount_latest': 0,
            'label': 'Receive Payment',
        },
        {
            CASE_ID_KEY: 'Case003',
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 1,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'Customer_latest': 'CustomerA',
            'Amount_latest': 300,
            'label': 'Order Returned',
        },
        {
            CASE_ID_KEY: 'Case003',
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 1,
            'Contact Supplier': 0,
            'Order Returned': 1,
            'Issue Refund': 0,
            'Customer_latest': 'CustomerA',
            'Amount_latest': 0,
            'label': 'Issue Refund',
        },
    ]

@pytest.fixture
def gt_encoded_log_prefix_length_up_to_2():
    return [
        # Case001
        {
            CASE_ID_KEY: 'Case001',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case001',
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
            CASE_ID_KEY: 'Case002',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Contact Supplier',
        },
        {
            CASE_ID_KEY: 'Case002',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 1,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Ship',
        },
        # Case003
        {
            CASE_ID_KEY: 'Case003',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Ship',
        },
        {
            CASE_ID_KEY: 'Case003',
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Receive Payment',
        },
    ]

@pytest.fixture
def gt_encoded_log_prefix_length_only_2():
    return [
        # Case001
        {
            CASE_ID_KEY: 'Case001',
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
            CASE_ID_KEY: 'Case002',
            'Receive Order': 1,
            'Ship': 0,
            'Receive Payment': 0,
            'Contact Supplier': 1,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Ship',
        },
        # Case003
        {
            CASE_ID_KEY: 'Case003',
            'Receive Order': 1,
            'Ship': 1,
            'Receive Payment': 0,
            'Contact Supplier': 0,
            'Order Returned': 0,
            'Issue Refund': 0,
            'label': 'Receive Payment',
        },
    ]


def test_frequency_encoder(log, gt_encoded_log):
    frequency_encoder = FrequencyEncoder(
        labeling_type=LabelingType.NEXT_ACTIVITY,
        timestamp_format=TIMESTAMP_FORMAT,
        case_id_key=CASE_ID_KEY,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
    )
    encoded_log = frequency_encoder.encode(log)

    assert len(gt_encoded_log) == len(encoded_log)
    assert len(encoded_log.columns) == NUM_ACTIVITIES + 1 + 1 # + 1 is case id, + 1 is label

    encoded_log = encoded_log.to_dict(orient='records')
    for i in range(len(gt_encoded_log)):
        assert gt_encoded_log[i] == encoded_log[i]


def test_frequency_encoder_latest_payload(log, gt_encoded_log_latest_payload):
    frequency_encoder = FrequencyEncoder(
        labeling_type=LabelingType.NEXT_ACTIVITY,
        timestamp_format=TIMESTAMP_FORMAT,
        case_id_key=CASE_ID_KEY,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
    )
    encoded_log = frequency_encoder.encode(
        log,
        include_latest_payload=True,
        attributes='all',
        categorical_attributes_encoding=CategoricalEncoding.STRING,
    )

    assert len(gt_encoded_log_latest_payload) == len(encoded_log)
    assert len(encoded_log.columns) == NUM_ACTIVITIES + 1 + 1 + 2 # + 1 is case id, + 1 is label, + 2 are attributes

    encoded_log = encoded_log.to_dict(orient='records')
    for i in range(len(gt_encoded_log_latest_payload)):
        assert gt_encoded_log_latest_payload[i] == encoded_log[i]


def test_frequency_encoder_prefix_length_up_to_2(log, gt_encoded_log_prefix_length_up_to_2):
    frequency_encoder = FrequencyEncoder(
        labeling_type=LabelingType.NEXT_ACTIVITY,
        prefix_length=2,
        prefix_strategy=PrefixStrategy.UP_TO_SPECIFIED,
        timestamp_format=TIMESTAMP_FORMAT,
        case_id_key=CASE_ID_KEY,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
    )
    encoded_log = frequency_encoder.encode(log)

    assert len(gt_encoded_log_prefix_length_up_to_2) == len(encoded_log)
    assert len(encoded_log.columns) == NUM_ACTIVITIES + 1 + 1 # + 1 is case id, + 1 is label

    encoded_log = encoded_log.to_dict(orient='records')
    for i in range(len(gt_encoded_log_prefix_length_up_to_2)):
        assert gt_encoded_log_prefix_length_up_to_2[i] == encoded_log[i]


def test_frequency_encoder_prefix_length_only_2(log, gt_encoded_log_prefix_length_only_2):
    frequency_encoder = FrequencyEncoder(
        labeling_type=LabelingType.NEXT_ACTIVITY,
        prefix_length=2,
        prefix_strategy=PrefixStrategy.ONLY_SPECIFIED,
        timestamp_format=TIMESTAMP_FORMAT,
        case_id_key=CASE_ID_KEY,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
    )
    encoded_log = frequency_encoder.encode(log)

    assert len(gt_encoded_log_prefix_length_only_2) == len(encoded_log)
    assert len(encoded_log.columns) == NUM_ACTIVITIES + 1 + 1 # + 1 is case id, + 1 is label

    encoded_log = encoded_log.to_dict(orient='records')
    for i in range(len(gt_encoded_log_prefix_length_only_2)):
        assert gt_encoded_log_prefix_length_only_2[i] == encoded_log[i]