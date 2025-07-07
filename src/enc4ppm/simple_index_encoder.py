import pandas as pd

from .base_encoder import BaseEncoder
from .constants import LabelingType

class SimpleIndexEncoder(BaseEncoder):
    PADDING_VALUE = 'PADDING'
    EVENT_COL_NAME = 'event'

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
    Encode the provided DataFrame with simple-index encoding and apply the specified labeling.

    Args:
        df: DataFrame to encode.
        labeling_type: Label type to apply to examples.

    Returns:
        The encoded DataFrame.
    """
    def encode(
        self,
        df: pd.DataFrame,
        *,
        labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY
    ) -> pd.DataFrame:
        df = super().validate_and_prepare_log(df, labeling_type=labeling_type)
        
        # Custom logic
        cases = df[self.case_id_key].unique().tolist()
        max_prefix_length = df.groupby(self.case_id_key).size().max()

        encoded_df = pd.DataFrame(
            columns=[self.ORIGINAL_CASE_ID_KEY, self.ORIGINAL_INDEX_KEY] +
                [f'{self.EVENT_COL_NAME}_{i}' for i in range(1, max_prefix_length+1)]
        )

        for prefix_length in range(1, max_prefix_length+1):
            for case in cases:
                case_events = df[df[self.case_id_key] == case]
                if len(case_events) < prefix_length: continue
                case_events = case_events.iloc[:prefix_length]

                encoded_row = {
                    self.ORIGINAL_CASE_ID_KEY: case,
                    self.ORIGINAL_INDEX_KEY: case_events.index[prefix_length-1],
                    **{ f'{self.EVENT_COL_NAME}_{i}': self.PADDING_VALUE for i in range(1, max_prefix_length+1) },
                }

                case_events = case_events.reset_index(drop=True)

                for i, case_event in case_events.iterrows():
                    encoded_row[f'{self.EVENT_COL_NAME}_{i+1}'] = case_event[self.activity_key]

                encoded_df = pd.concat([encoded_df, pd.DataFrame([encoded_row])], ignore_index=True)

        encoded_df = self.label_log(encoded_df, labeling_type=labeling_type)

        return encoded_df