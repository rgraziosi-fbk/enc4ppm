from abc import ABC, abstractmethod
import pandas as pd

class BaseEncoder(ABC):
    def __init__(
            self,
            case_id_key: str = 'case:concept:name',
            activity_key: str = 'concept:name',
            timestamp_key: str = 'time:timestamp') -> None:
        self.case_id_key = case_id_key
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key

    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def validate_and_prepare_log(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty")
        
        for col in [self.case_id_key, self.activity_key, self.timestamp_key]:
            if col not in df.columns:
                raise ValueError(f"Input DataFrame must contain column '{col}'")
            
        df[self.timestamp_key] = pd.to_datetime(df[self.timestamp_key])

        return df
