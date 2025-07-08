import pandas as pd

from .base_encoder import BaseEncoder
from .constants import LabelingType

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
        activities = df[self.activity_key].unique().tolist()

        rows = []
        
        for case_id, case_events in grouped:
            case_events = case_events.sort_values(self.timestamp_key)

            for prefix_length in range(1, len(case_events)+1):
                prefix = case_events.iloc[:prefix_length]
                counts = prefix[self.activity_key].value_counts()

                row = {
                    self.ORIGINAL_CASE_ID_KEY: case_id,
                    self.ORIGINAL_INDEX_KEY: prefix.index[-1],
                }

                for activity in activities:
                    row[activity] = counts.get(activity, 0)

                rows.append(row)

        encoded_df = pd.DataFrame(rows)
        
        encoded_df = super()._label_log(encoded_df, labeling_type=labeling_type)
        encoded_df = super()._postprocess_log(encoded_df)

        return encoded_df
