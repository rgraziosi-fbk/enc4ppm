import pandas as pd
from pandas.api.types import is_object_dtype

from .base_encoder import BaseEncoder
from .constants import LabelingType, CategoricalEncoding, PrefixStrategy

class ComplexIndexEncoder(BaseEncoder):
    PADDING_VALUE = 'PADDING'
    EVENT_COL_NAME = 'event'

    def __init__(
        self,
        *,
        labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY,
        prefix_length: int = None,
        prefix_strategy: PrefixStrategy = PrefixStrategy.UP_TO_SPECIFIED,
        timestamp_format: str = None,
        case_id_key: str = 'case:concept:name',
        activity_key: str = 'concept:name',
        timestamp_key: str = 'time:timestamp',
    ) -> None:
        """
        Initialize the ComplexIndexEncoder.

        Args:
            labeling_type: Label type to apply to examples.
            prefix_length: Maximum prefix length to consider: longer prefixes will be discarded, shorter prefixes may be discarded depending on prefix_strategy parameter. If not provided, defaults to maximum prefix length found in log. If provided, it must be a non-zero positive int number.
            prefix_strategy: Whether to consider prefix lengths from 1 to prefix_length (PrefixStrategy.UP_TO_SPECIFIED) or only the specified prefix_length (PrefixStrategy.ONLY_SPECIFIED).
            timestamp_format: Format of the timestamps in the log. If not provided, formatting will be inferred from the data.
            case_id_key: Column name for case identifiers.
            activity_key: Column name for activity names.
            timestamp_key: Column name for timestamps.
        """
        super().__init__(
            labeling_type,
            prefix_length,
            prefix_strategy,
            timestamp_format,
            case_id_key,
            activity_key,
            timestamp_key,
        )


    def encode(
        self,
        df: pd.DataFrame,
        *,
        activity_encoding: CategoricalEncoding = CategoricalEncoding.STRING,
        attributes: str | list = 'all',
        categorical_attributes_encoding: CategoricalEncoding = CategoricalEncoding.STRING,
    ) -> pd.DataFrame:
        """
        Encode the provided DataFrame with complex-index encoding and apply the specified labeling.

        Args:
            df: DataFrame to encode.
            activity_encoding: How to encode activity names. They can either remain strings (CategoricalEncoding.STRING) or be converted to one-hot vectors splitted across multiple columns (CategoricalEncoding.ONE_HOT).
            attributes: Which attributes to consider. Can be either 'all' (all trace and event attributes will be encoded) or a list of the attributes to consider.
            categorical_attributes_encoding: How to encode categorical attributes. They can either remain strings (CategoricalEncoding.STRING) or be converted to one-hot vectors splitted across multiple columns (CategoricalEncoding.ONE_HOT).

        Returns:
            The encoded DataFrame.
        """
        return super()._encode_template(
            df,
            activity_encoding=activity_encoding,
            attributes=attributes,
            categorical_attributes_encoding=categorical_attributes_encoding,
        )
    

    def _encode(
        self,
        df: pd.DataFrame,
        activity_encoding: CategoricalEncoding,
        attributes: str | list = 'all',
        categorical_attributes_encoding: CategoricalEncoding = CategoricalEncoding.STRING,
    ) -> pd.DataFrame:
        grouped = df.groupby(self.case_id_key)
        max_prefix_length = grouped.size().max()

        rows = []

        # If attributes set to 'all', obtain all available attributes from dataframe
        if attributes == 'all':
            attributes = [a for a in self.original_df.columns.tolist() if a not in [self.case_id_key, self.activity_key, self.timestamp_key]]

        for case_id, case_events in grouped:
            case_events = case_events.sort_values(self.timestamp_key).reset_index()

            for prefix_length in range(1, len(case_events)+1):
                row = {
                    self.case_id_key: case_id,
                    self.timestamp_key: case_events.loc[prefix_length-1, self.timestamp_key],
                    self.ORIGINAL_INDEX_KEY: case_events.loc[prefix_length-1, 'index'],
                }

                for i in range(1, min(self.prefix_length, max_prefix_length)+1):
                    if i <= prefix_length:
                        row[f'{self.EVENT_COL_NAME}_{i}'] = case_events.loc[i-1, self.activity_key]
                    else:
                        row[f'{self.EVENT_COL_NAME}_{i}'] = self.PADDING_VALUE

                    for attribute in attributes:
                        if i <= prefix_length:
                            row[f'{attribute}_{i}'] = case_events.loc[i-1, attribute]
                        else:
                            row[f'{attribute}_{i}'] = self.PADDING_VALUE
                
                rows.append(row)

        encoded_df = pd.DataFrame(rows)

        # Transform activities to one-hot if requested
        if activity_encoding == CategoricalEncoding.ONE_HOT:
            encoded_df = pd.get_dummies(
                encoded_df,
                columns=[f'{self.EVENT_COL_NAME}_{i}' for i in range(1, max_prefix_length+1)],
                drop_first=True,
            )

        # Transform attributes to one-hot if requested
        if categorical_attributes_encoding == CategoricalEncoding.ONE_HOT:
            categorical_columns = []
            
            for attribute in attributes:
                if is_object_dtype(df[f'{attribute}_{self.LATEST_PAYLOAD_COL_SUFFIX_NAME}']):
                    categorical_columns.append(f'{attribute}_{self.LATEST_PAYLOAD_COL_SUFFIX_NAME}')

            encoded_df = pd.get_dummies(
                encoded_df,
                columns=categorical_columns,
                drop_first=True,
            )

        return encoded_df
