import pandas as pd
from pandas.api.types import is_object_dtype

from .base_encoder import BaseEncoder
from .constants import LabelingType, CategoricalEncoding

class FrequencyEncoder(BaseEncoder):
    """
    Initialize the FrequencyEncoder.

    Args:
        case_id_key: Column name for case identifiers.
        activity_key: Column name for activity names.
        timestamp_key: Column name for timestamps.
    """
    def __init__(
            self,
            *,
            case_id_key: str = 'case:concept:name',
            activity_key: str = 'concept:name',
            timestamp_key: str = 'time:timestamp') -> None:
        super().__init__(case_id_key, activity_key, timestamp_key)

    """
    Encode the provided DataFrame with frequency encoding and apply the specified labeling.

    Args:
        df: DataFrame to encode.
        labeling_type: Label type to apply to examples.
        include_latest_payload: Whether to include (True) or not (False) the latest values of trace and event attributes. The attributes to consider can be specified through the `attributes` parameter.
        attributes: Which attributes to consider. Can be either 'all' (all trace and event attributes will be encoded) or a list of the attributes to consider.
        categorical_attributes_encoding: How to encode categorical attributes. They can either remain strings (CategoricalEncoding.STRING) or be converted to one-hot vectors splitted across multiple columns (CategoricalEncoding.ONE_HOT).

    Returns:
        The encoded DataFrame.
    """
    def encode(
        self,
        df: pd.DataFrame,
        *,
        labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY,
        include_latest_payload: bool = False,
        attributes: str | list = 'all',
        categorical_attributes_encoding: CategoricalEncoding = CategoricalEncoding.STRING,
    ) -> pd.DataFrame:
        df = super()._preprocess_log(df, labeling_type=labeling_type)

        # TODO: remove duplication (see simple_index_encoder.py)
        if include_latest_payload and attributes == 'all':
            attributes = [a for a in self.original_df.columns.tolist() if a not in [self.case_id_key, self.activity_key, self.timestamp_key]]
        
        grouped = df.groupby(self.case_id_key)
        activities = df[self.activity_key].unique().tolist()

        rows = []
        
        for case_id, case_events in grouped:
            case_events = case_events.sort_values(self.timestamp_key)

            for prefix_length in range(1, len(case_events)+1):
                prefix = case_events.iloc[:prefix_length]
                counts = prefix[self.activity_key].value_counts()

                row = {
                    self.case_id_key: case_id,
                    self.ORIGINAL_INDEX_KEY: prefix.index[-1],
                }

                for activity in activities:
                    row[activity] = counts.get(activity, 0)

                rows.append(row)

        encoded_df = pd.DataFrame(rows)
        
        if include_latest_payload:
            encoded_df = super()._include_latest_payload(encoded_df, attributes=attributes)

            if categorical_attributes_encoding == CategoricalEncoding.ONE_HOT:
                encoded_df = pd.get_dummies(
                    encoded_df,
                    columns=[f'{attribute}_latest' for attribute in attributes if is_object_dtype(encoded_df[f'{attribute}_latest'])],
                    drop_first=True,
                )

        encoded_df = super()._label_log(encoded_df, labeling_type=labeling_type)
        encoded_df = super()._postprocess_log(encoded_df)

        return encoded_df
