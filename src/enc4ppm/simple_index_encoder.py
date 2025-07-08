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
        df = super()._preprocess_log(df, labeling_type=labeling_type)
        
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

        encoded_df = super()._label_log(encoded_df, labeling_type=labeling_type)
        encoded_df = super()._postprocess_log(encoded_df)

        return encoded_df