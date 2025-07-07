from abc import ABC, abstractmethod
import pandas as pd

from .constants import LabelingType

class BaseEncoder(ABC):
    ORIGINAL_CASE_ID_KEY = 'OriginalCaseId'
    ORIGINAL_INDEX_KEY = 'OriginalIndex'
    LABEL_KEY = 'label'
    
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

    def validate_and_prepare_log(self, df: pd.DataFrame, labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty")
        
        for col in [self.case_id_key, self.activity_key, self.timestamp_key]:
            if col not in df.columns:
                raise ValueError(f"Input DataFrame must contain column '{col}'")
            
        if not isinstance(labeling_type, LabelingType):
            raise TypeError(f"labeling_type must be a valid LabelingType: {[e.name for e in LabelingType]}")
            
        # Cast timestamp column to datetime
        df[self.timestamp_key] = pd.to_datetime(df[self.timestamp_key])

        # Sort by start timestamp
        # TODO: bugfix: if two cases start at the same time, then their events interleave in the sorted dataframe
        df['_first_timestamp'] = df.groupby(self.case_id_key)[self.timestamp_key].transform('min')
        df = df.sort_values(by=['_first_timestamp', self.case_id_key, self.timestamp_key]).reset_index(drop=True)
        df = df.drop(columns=['_first_timestamp'])

        # Save original df for later use
        self.original_df = df

        return df

    def label_log(self, df: pd.DataFrame, labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY) -> pd.DataFrame:
        if labeling_type == LabelingType.NEXT_ACTIVITY:
            labels = []
            for _, row in df.iterrows():
                same_case = df[
                    (df[self.ORIGINAL_CASE_ID_KEY] == row[self.ORIGINAL_CASE_ID_KEY]) &
                    (df[self.ORIGINAL_INDEX_KEY] > row[self.ORIGINAL_INDEX_KEY])
                ]
                if not same_case.empty:
                    next_index = same_case[self.ORIGINAL_INDEX_KEY].min()
                    label = self.original_df.loc[next_index, self.activity_key]
                else:
                    label = None
                labels.append(label)
            df[self.LABEL_KEY] = labels

        # Restore the original ordering
        df = df.sort_values(by=self.ORIGINAL_INDEX_KEY).reset_index(drop=True)

        # Drop unnecessary data
        df = df.drop(columns=[self.ORIGINAL_CASE_ID_KEY, self.ORIGINAL_INDEX_KEY])
        df = df.dropna(subset=[self.LABEL_KEY]).reset_index(drop=True)

        return df
