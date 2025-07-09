from abc import ABC, abstractmethod
import pandas as pd

from .constants import LabelingType

class BaseEncoder(ABC):
    ORIGINAL_INDEX_KEY = 'OriginalIndex'
    LABEL_KEY = 'label'
    
    def __init__(
            self,
            labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY,
            case_id_key: str = 'case:concept:name',
            activity_key: str = 'concept:name',
            timestamp_key: str = 'time:timestamp',
        ) -> None:
        self.labeling_type = labeling_type
        self.case_id_key = case_id_key
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key

    """
    The _encode abstract method must be defined by subclasses and must contain the specific encoding logic of the encoder.
    """
    @abstractmethod
    def _encode(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

    """
    The _encode_template method is a template method which performs both common operations shared amongs all encoders and the specific logic of each encoder.
    In particular, common operations are: _preprocess_log, _label_log and _postprocess_log; specific encoding is performed by subclass _encode method.
    """
    def _encode_template(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = self._preprocess_log(df, labeling_type=self.labeling_type)

        encoded_df = self._encode(df, **kwargs)

        encoded_df = self._label_log(encoded_df, labeling_type=self.labeling_type)
        encoded_df = self._postprocess_log(encoded_df)

        return encoded_df
    
    """
    Common preprocessing logic shared by all encoders. The method validates the provided log and save it for later use.
    """
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

    """
    Common labeling logic shared by all encoders. The method labels each example of the encoded log.
    """
    def _label_log(self, df: pd.DataFrame, labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY) -> pd.DataFrame:
        if self.ORIGINAL_INDEX_KEY not in df.columns:
            raise ValueError(f'You must include {self.ORIGINAL_INDEX_KEY} column into df before calling _label_log')
        
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
    
    """
    Common postprocessing logic shared by all encoders. The method restores original ordering and drops unnecessary data.
    """
    def _postprocess_log(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.ORIGINAL_INDEX_KEY not in df.columns:
            raise ValueError(f'You must include {self.ORIGINAL_INDEX_KEY} column into df before calling _postprocess_log')
        
        # Restore original ordering
        df = df.sort_values(by=self.ORIGINAL_INDEX_KEY).reset_index(drop=True)

        # Drop unnecessary data
        df = df.drop(columns=[self.ORIGINAL_INDEX_KEY])
        df = df.dropna(subset=[self.LABEL_KEY]).reset_index(drop=True)

        return df

    """
    Add to an already encoded 
    """
    def _include_latest_payload(self, df: pd.DataFrame, attributes: str | list = 'all'):
        if attributes == None: return df
        if self.ORIGINAL_INDEX_KEY not in df.columns:
            raise ValueError(f'You must include {self.ORIGINAL_INDEX_KEY} column into df before calling _include_latest_payload')

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