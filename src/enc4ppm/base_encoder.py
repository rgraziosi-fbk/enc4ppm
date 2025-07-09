from abc import ABC, abstractmethod
import pandas as pd

from .constants import LabelingType

class BaseEncoder(ABC):
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

    def _preprocess_log(self, df: pd.DataFrame, labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("df cannot be empty")
        
        for col in [self.case_id_key, self.activity_key, self.timestamp_key]:
            if col not in df.columns:
                raise ValueError(f"df must contain column '{col}'")
            
        if not isinstance(labeling_type, LabelingType):
            raise TypeError(f'labeling_type must be a valid LabelingType: {[e.name for e in LabelingType]}')
            
        # Cast timestamp column to datetime
        df[self.timestamp_key] = pd.to_datetime(df[self.timestamp_key])

        # Save original df for later use
        self.original_df = df

        return df

    def _include_latest_payload(self, df: pd.DataFrame, attributes: str | list = 'all'):
        if attributes == None: return df

        if isinstance(attributes, str) and attributes != 'all':
            raise ValueError("Since attributes is set to a string, then it must be set to the value 'all'. Otherwise, set it to a list of strings indicating the attributes you want to consider.")
        
        if isinstance(attributes, list):
            for payload_attribute in attributes:
                if not isinstance(payload_attribute, str):
                    raise ValueError('Since attributes is a list, it must contain only string elements')
                
                if payload_attribute not in self.original_df.columns:
                    raise ValueError(f"attributes contains value '{payload_attribute}', which cannot be found in the log")
        
        # If attributes set to 'all', obtain all available attributes from dataframe
        if attributes == 'all':
            attributes = [a for a in self.original_df.columns.tolist() if a not in [self.case_id_key, self.activity_key, self.timestamp_key]]

        for payload_attribute in attributes:
            attribute_values = []
            
            for _, row in df.iterrows():
                attribute_values.append(self.original_df.loc[row[self.ORIGINAL_INDEX_KEY], payload_attribute])

            df[f'{payload_attribute}_latest'] = attribute_values

        return df, attributes

    def _label_log(self, df: pd.DataFrame, labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY) -> pd.DataFrame:
        if labeling_type == LabelingType.NEXT_ACTIVITY:
            labels = []
            
            for _, row in df.iterrows():
                same_case = df[
                    (df[self.case_id_key] == row[self.case_id_key]) &
                    (df[self.ORIGINAL_INDEX_KEY] > row[self.ORIGINAL_INDEX_KEY])
                ]
                if not same_case.empty:
                    next_index = same_case[self.ORIGINAL_INDEX_KEY].min()
                    label = self.original_df.loc[next_index, self.activity_key]
                else:
                    label = None
                labels.append(label)
            
            df[self.LABEL_KEY] = labels

        return df
    
    def _postprocess_log(self, df: pd.DataFrame) -> pd.DataFrame:
        # Restore original ordering
        df = df.sort_values(by=self.ORIGINAL_INDEX_KEY).reset_index(drop=True)

        # Drop unnecessary data
        df = df.drop(columns=[self.ORIGINAL_INDEX_KEY])
        df = df.dropna(subset=[self.LABEL_KEY]).reset_index(drop=True)

        return df
