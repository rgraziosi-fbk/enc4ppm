from .base_encoder import BaseEncoder
import pandas as pd

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
            case_id_key: str = 'case:concept:name',
            activity_key: str = 'concept:name',
            timestamp_key: str = 'time:timestamp') -> None:
        super().__init__(case_id_key, activity_key, timestamp_key)

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        # BaseEncoder logic
        df = super().validate_and_prepare_log(df)
        
        # Custom logic
        activities = df[self.activity_key].unique().tolist()
        cases = df[self.case_id_key].unique().tolist()
        max_prefix_length = df.groupby(self.case_id_key).size().max()
        
        encoded_df = pd.DataFrame(columns=[self.case_id_key] + activities)

        for prefix_length in range(1, max_prefix_length+1):
            for case in cases:
                case_events = df[df[self.case_id_key] == case]
                if len(case_events) < prefix_length: continue
                case_events = case_events.iloc[:prefix_length]

                encoded_row = { self.case_id_key: case, **{ activity: 0 for activity in activities } }
                
                for activity in activities:
                    activity_events = case_events[case_events[self.activity_key] == activity]
                    encoded_row[activity] = activity_events.shape[0]

                encoded_df = pd.concat([encoded_df, pd.DataFrame([encoded_row])], ignore_index=True)

        return encoded_df
