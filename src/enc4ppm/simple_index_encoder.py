import pandas as pd

from .base_encoder import BaseEncoder
from .constants import LabelingType, CategoricalEncoding

class SimpleIndexEncoder(BaseEncoder):
    PADDING_VALUE = 'PADDING'
    EVENT_COL_NAME = 'event'

    def __init__(
        self,
        *,
        labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY,
        case_id_key: str = 'case:concept:name',
        activity_key: str = 'concept:name',
        timestamp_key: str = 'time:timestamp',
    ) -> None:
        """
        Initialize the SimpleIndexEncoder.

        Args:
            labeling_type: Label type to apply to examples.
            case_id_key: Column name for case identifiers.
            activity_key: Column name for activity names.
            timestamp_key: Column name for timestamps.
        """
        super().__init__(labeling_type, case_id_key, activity_key, timestamp_key)

    
    def encode(
        self,
        df: pd.DataFrame,
        *,
        activity_encoding: CategoricalEncoding = CategoricalEncoding.STRING,
        include_latest_payload: bool = False,
        attributes: str | list = 'all',
        categorical_attributes_encoding: CategoricalEncoding = CategoricalEncoding.STRING,
    ) -> pd.DataFrame:
        """
        Encode the provided DataFrame with simple-index encoding and apply the specified labeling.

        Args:
            df: DataFrame to encode.
            activity_encoding: How to encode activity names. They can either remain strings (CategoricalEncoding.STRING) or be converted to one-hot vectors splitted across multiple columns (CategoricalEncoding.ONE_HOT).
            include_latest_payload: Whether to include (True) or not (False) the latest values of trace and event attributes. The attributes to consider can be specified through the `attributes` parameter.
            attributes: Which attributes to consider. Can be either 'all' (all trace and event attributes will be encoded) or a list of the attributes to consider.
            categorical_attributes_encoding: How to encode categorical attributes. They can either remain strings (CategoricalEncoding.STRING) or be converted to one-hot vectors splitted across multiple columns (CategoricalEncoding.ONE_HOT).

        Returns:
            The encoded DataFrame.
        """
        return super()._encode_template(
            df,
            activity_encoding=activity_encoding,
            include_latest_payload=include_latest_payload,
            attributes=attributes,
            categorical_attributes_encoding=categorical_attributes_encoding,
        )


    def _encode(
        self,
        df: pd.DataFrame,
        activity_encoding: CategoricalEncoding,
        include_latest_payload: bool = False,
        attributes: str | list = 'all',
        categorical_attributes_encoding: CategoricalEncoding = CategoricalEncoding.STRING,
    ) -> pd.DataFrame:
        grouped = df.groupby(self.case_id_key)
        max_prefix_length = grouped.size().max()

        rows = []

        for case_id, case_events in grouped:
            case_events = case_events.sort_values(self.timestamp_key).reset_index()

            for prefix_length in range(1, len(case_events)+1):
                row = {
                    self.case_id_key: case_id,
                    self.ORIGINAL_INDEX_KEY: case_events.loc[prefix_length-1, 'index'],
                }

                for i in range(1, max_prefix_length+1):
                    if i <= prefix_length:
                        row[f'{self.EVENT_COL_NAME}_{i}'] = case_events.loc[i-1, self.activity_key]
                    else:
                        row[f'{self.EVENT_COL_NAME}_{i}'] = self.PADDING_VALUE
                
                rows.append(row)

        encoded_df = pd.DataFrame(rows)

        if activity_encoding == CategoricalEncoding.ONE_HOT:
            encoded_df = pd.get_dummies(
                encoded_df,
                columns=[f'{self.EVENT_COL_NAME}_{i}' for i in range(1, max_prefix_length+1)],
                drop_first=True,
            )

        if include_latest_payload:
            encoded_df = super()._include_latest_payload(
                encoded_df,
                attributes=attributes,
                categorical_attributes_encoding=categorical_attributes_encoding
            )

        return encoded_df
