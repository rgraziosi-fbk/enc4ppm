import pandas as pd

from .base_encoder import BaseEncoder
from .constants import LabelingType, CategoricalEncoding, PrefixStrategy

class FrequencyEncoder(BaseEncoder):
    include_latest_payload: bool = None
    attributes: str | list[str] = None
    categorical_attributes_encoding: CategoricalEncoding = None

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
        outcome_key: str = 'outcome',
    ) -> None:
        """
        Initialize the FrequencyEncoder.

        Args:
            labeling_type: Label type to apply to examples.
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
        include_latest_payload: bool = False,
        attributes: str | list[str] = 'all',
        categorical_attributes_encoding: CategoricalEncoding = CategoricalEncoding.STRING,
    ) -> pd.DataFrame:
        """
        Encode the provided DataFrame with frequency encoding and apply the specified labeling.

        Args:
            df: DataFrame to encode.
            freeze: Freeze encoder with provided parameters. Usually set to True when encoding the train log, False otherwise. Required if you want to later save the encoder to a file.
            include_latest_payload: Whether to include (True) or not (False) the latest values of trace and event attributes. The attributes to consider can be specified through the `attributes` parameter.
            attributes: Which attributes to consider. Can be either 'all' (all trace and event attributes will be encoded) or a list of the attributes to consider.
            categorical_attributes_encoding: How to encode categorical attributes. They can either remain strings (CategoricalEncoding.STRING) or be converted to one-hot vectors splitted across multiple columns (CategoricalEncoding.ONE_HOT).

        Returns:
            The encoded DataFrame.
        """
        return super()._encode_template(
            df,
            freeze=freeze,
            include_latest_payload=include_latest_payload,
            attributes=attributes,
            categorical_attributes_encoding=categorical_attributes_encoding,
        )


    def _encode(
        self,
        df: pd.DataFrame,
        freeze: bool,
        include_latest_payload: bool,
        attributes: str | list,
        categorical_attributes_encoding: CategoricalEncoding,
    ) -> pd.DataFrame:
        if freeze:
            self.include_latest_payload = include_latest_payload
            self.attributes = attributes
            self.categorical_attributes_encoding = categorical_attributes_encoding

        if self.is_frozen:
            include_latest_payload = self.include_latest_payload
            attributes = self.attributes
            categorical_attributes_encoding = self.categorical_attributes_encoding

        grouped = df.groupby(self.case_id_key)

        rows = []
        
        for case_id, case_events in grouped:
            case_events = case_events.sort_values(self.timestamp_key)

            for prefix_length in range(1, len(case_events)+1):
                prefix = case_events.iloc[:prefix_length]
                counts = prefix[self.activity_key].value_counts()

                row = {
                    self.case_id_key: case_id,
                    self.timestamp_key: prefix.iloc[-1][self.timestamp_key],
                    self.ORIGINAL_INDEX_KEY: prefix.index[-1],
                }

                for activity in self.log_activities[:-1]:
                    row[activity] = counts.get(activity, 0)
                
                # Handle unknown activities
                row[self.UNKNOWN_VAL] = sum(counts.get(activity, 0) for activity in counts.index if activity not in self.log_activities[:-1])

                rows.append(row)

        encoded_df = pd.DataFrame(rows)
        
        if include_latest_payload:
            encoded_df = super()._include_latest_payload(
                encoded_df,
                attributes=attributes,
                categorical_attributes_encoding=categorical_attributes_encoding
            )

        return encoded_df
