import pandas as pd
from pandas.api.types import is_object_dtype

from .base_encoder import BaseEncoder
from .constants import LabelingType, CategoricalEncoding, PrefixStrategy

class ComplexIndexEncoder(BaseEncoder):
    PADDING_CAT_VALUE = 'PADDING'
    PADDING_NUM_VALUE = 0.0
    EVENT_COL_NAME = 'event'

    def __init__(
        self,
        *,
        labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY,
        attributes: list[str] | str = [],
        categorical_encoding: CategoricalEncoding = CategoricalEncoding.STRING,
        prefix_length: int = None,
        prefix_strategy: PrefixStrategy = PrefixStrategy.UP_TO_SPECIFIED,
        timestamp_format: str = None,
        case_id_key: str = 'case:concept:name',
        activity_key: str = 'concept:name',
        timestamp_key: str = 'time:timestamp',
        outcome_key: str = 'outcome',
    ) -> None:
        """
        Initialize the ComplexIndexEncoder.

        Args:
            labeling_type: Label type to apply to examples.
            attributes: Which attributes to consider. Can be a list of the attributes to consider or the string 'all' (all attributes found in the log will be encoded).
            categorical_attributes_encoding: How to encode categorical attributes. They can either remain strings (CategoricalEncoding.STRING) or be converted to one-hot vectors splitted across multiple columns (CategoricalEncoding.ONE_HOT).
            prefix_length: Maximum prefix length to consider: longer prefixes will be discarded, shorter prefixes may be discarded depending on prefix_strategy parameter. If not provided, defaults to maximum prefix length found in log. If provided, it must be a non-zero positive int number.
            prefix_strategy: Whether to consider prefix lengths from 1 to prefix_length (PrefixStrategy.UP_TO_SPECIFIED) or only the specified prefix_length (PrefixStrategy.ONLY_SPECIFIED).
            timestamp_format: Format of the timestamps in the log. If not provided, formatting will be inferred from the data.
            case_id_key: Column name for case identifiers.
            activity_key: Column name for activity names.
            timestamp_key: Column name for timestamps.
            outcome_key: Column name for outcome predition.
        """
        super().__init__(
            labeling_type,
            attributes,
            categorical_encoding,
            prefix_length,
            prefix_strategy,
            timestamp_format,
            case_id_key,
            activity_key,
            timestamp_key,
            outcome_key,
        )


    def encode(
        self,
        df: pd.DataFrame,
        *,
        freeze: bool = False,
    ) -> pd.DataFrame:
        """
        Encode the provided DataFrame with complex-index encoding and apply the specified labeling.

        Args:
            df: DataFrame to encode.
            freeze: Freeze encoder with provided parameters. Usually set to True when encoding the train log, False otherwise. Required if you want to later save the encoder to a file.

        Returns:
            The encoded DataFrame.
        """
        return super()._encode_template(df, freeze=freeze)
    

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby(self.case_id_key)
        max_prefix_length = grouped.size().max()

        rows = []

        # Build a dictionary mapping dynamic_attributes to either 'cat' or 'num', based on whether the attributes are categorical or numerical
        dynamic_attributes_types = {
            attr: 'cat' if is_object_dtype(df[attr]) else 'num' for attr in dynamic_attributes
        }

        for case_id, case_events in grouped:
            case_events = case_events.sort_values(self.timestamp_key).reset_index()

            for prefix_length in range(1, len(case_events)+1):
                row = {
                    self.case_id_key: case_id,
                    self.timestamp_key: case_events.loc[prefix_length-1, self.timestamp_key],
                    self.ORIGINAL_INDEX_KEY: case_events.loc[prefix_length-1, 'index'],
                }

                # Add static attributes
                for static_attribute in static_attributes:
                    row[static_attribute] = case_events.loc[prefix_length-1, static_attribute]

                for i in range(1, min(self.prefix_length, max_prefix_length)+1):
                    # Add activities
                    if i <= prefix_length:
                        row[f'{self.EVENT_COL_NAME}_{i}'] = case_events.loc[i-1, self.activity_key]
                    else:
                        row[f'{self.EVENT_COL_NAME}_{i}'] = self.PADDING_CAT_VALUE

                    # Add dynamic attributes
                    for dynamic_attribute in dynamic_attributes:
                        if i <= prefix_length:
                            row[f'{dynamic_attribute}_{i}'] = case_events.loc[i-1, dynamic_attribute]
                        else:
                            row[f'{dynamic_attribute}_{i}'] = dynamic_attributes_types[dynamic_attribute] == 'cat' and self.PADDING_CAT_VALUE or self.PADDING_NUM_VALUE
                
                rows.append(row)

        encoded_df = pd.DataFrame(rows)

        # Transform activities to one-hot if requested
        if self.categorical_encoding == CategoricalEncoding.ONE_HOT:
            encoded_df = pd.get_dummies(
                encoded_df,
                columns=[f'{self.EVENT_COL_NAME}_{i}' for i in range(1, min(self.prefix_length, max_prefix_length)+1)],
                drop_first=True,
            )

        # Transform attributes to one-hot if requested
        if self.categorical_encoding == CategoricalEncoding.ONE_HOT:
            categorical_columns = []
            
            for static_attribute in static_attributes:
                if is_object_dtype(encoded_df[static_attribute]):
                    categorical_columns.append(static_attribute)

            for dynamic_attribute in dynamic_attributes:
                for i in range(1, min(self.prefix_length, max_prefix_length)+1):
                    if is_object_dtype(encoded_df[f'{dynamic_attribute}_{i}']):
                        categorical_columns.append(f'{dynamic_attribute}_{i}')

            encoded_df = pd.get_dummies(
                encoded_df,
                columns=categorical_columns,
                drop_first=True,
            )

        return encoded_df
