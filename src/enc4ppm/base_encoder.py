from abc import ABC, abstractmethod
import pandas as pd
from pandas.api.types import is_object_dtype

from .constants import LabelingType, CategoricalEncoding, PrefixStrategy

class BaseEncoder(ABC):
    ORIGINAL_INDEX_KEY = 'OriginalIndex'
    LABEL_KEY = 'label'
    LATEST_PAYLOAD_COL_SUFFIX_NAME = 'latest'
    
    def __init__(
            self,
            labeling_type: LabelingType = LabelingType.NEXT_ACTIVITY,
            prefix_length: int = None,
            prefix_strategy: PrefixStrategy = PrefixStrategy.UP_TO_SPECIFIED,
            case_id_key: str = 'case:concept:name',
            activity_key: str = 'concept:name',
            timestamp_key: str = 'time:timestamp',
        ) -> None:
        self.labeling_type = labeling_type
        self.prefix_length = prefix_length
        self.prefix_strategy = prefix_strategy
        self.case_id_key = case_id_key
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key


    @abstractmethod
    def _encode(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        The _encode abstract method must be defined by subclasses and must contain the specific encoding logic of the encoder.
        In particular, the _encode implementation must create the necessary columns for the specific encoding + add the ORIGINAL_INDEX_KEY column.
        The _encode method must not filter rows (events), but instead return them all: the BaseEncoder will then _apply_prefix_strategy to filter them.
        """
        pass

    
    def _encode_template(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        The _encode_template method is a template method which performs both common operations shared amongs all encoders and the specific logic of each encoder.
        In particular, common operations are: _preprocess_log, _label_log, _apply_prefix_strategy and _postprocess_log; specific encoding is performed by the _encode method.
        """
        df = self._preprocess_log(df)

        encoded_df = self._encode(df, **kwargs)

        encoded_df = self._label_log(encoded_df)
        encoded_df = self._apply_prefix_strategy(encoded_df)
        encoded_df = self._postprocess_log(encoded_df)

        return encoded_df
    

    def _preprocess_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common preprocessing logic shared by all encoders. The method validates the provided log and save it for later use.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("df cannot be empty")
        
        for col in [self.case_id_key, self.activity_key, self.timestamp_key]:
            if col not in df.columns:
                raise ValueError(f"df must contain column '{col}'")
            
        if not isinstance(self.labeling_type, LabelingType):
            raise TypeError(f'labeling_type must be a valid LabelingType: {[e.name for e in LabelingType]}')
        
        if self.prefix_length is not None and (not isinstance(self.prefix_length, int) or self.prefix_length <= 0):
            raise ValueError(f'prefix_length must be either None or a positive integer ({self.prefix_length} has been provided instead)')
        
        if self.prefix_length is None and self.prefix_strategy == PrefixStrategy.ONLY_SPECIFIED:
            raise ValueError(f'If prefix strategy is set to ONLY_SPECIFIED, then you must specify the prefix_length parameter')
        
        if not isinstance(self.prefix_strategy, PrefixStrategy):
            raise TypeError(f'prefix_strategy must be a valid PrefixStrategy: {[e.name for e in PrefixStrategy]}')
            
        # Get prefix length to consider based both on provided one and maximum found in log
        max_prefix_length_log = df.groupby(self.case_id_key).size().max()
        
        if self.prefix_length is None:
            self.prefix_length = max_prefix_length_log
        else:
            if self.prefix_length > max_prefix_length_log:
                print(f'Warning: provided prefix_length {self.prefix_length} is higher than maximum prefix length found in log {max_prefix_length_log}! Setting prefix_length to {max_prefix_length_log}.')

            self.prefix_length = min(self.prefix_length, max_prefix_length_log)

        # Cast timestamp column to datetime
        df[self.timestamp_key] = pd.to_datetime(df[self.timestamp_key])

        # Save original df for later use
        self.original_df = df

        return df

    
    def _label_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common logic shared by all encoders. The method labels the provided log with the provided LabelingType.
        """
        if self.ORIGINAL_INDEX_KEY not in df.columns:
            raise ValueError(f'You must include {self.ORIGINAL_INDEX_KEY} column into df before calling _label_log')
        
        if self.labeling_type == LabelingType.NEXT_ACTIVITY:
            # Get the next ORIGINAL_INDEX_KEY per case
            df = df.sort_values([self.case_id_key, self.timestamp_key], ascending=[True, True]).reset_index(drop=True)
            df['next_index'] = df.groupby(self.case_id_key)[self.ORIGINAL_INDEX_KEY].shift(-1)

            # Map next_index to activity in original_df
            df[self.LABEL_KEY] = df['next_index'].map(self.original_df[self.activity_key])
            
            # Drop the helper column
            df = df.drop(columns=['next_index'])

        return df
    
    
    def _apply_prefix_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common logic shared by all encoders. The method filters the log with respect to specified prefix_length value.
        """
        # Compute event number in case (starting from 1)
        df = df.sort_values([self.case_id_key, self.timestamp_key], ascending=[True, True]).reset_index(drop=True)
        df['event_num_in_case'] = df.groupby(self.case_id_key).cumcount() + 1

        if self.prefix_strategy == PrefixStrategy.UP_TO_SPECIFIED:
            filtered_df = df[df['event_num_in_case'] <= self.prefix_length]
        elif self.prefix_strategy == PrefixStrategy.ONLY_SPECIFIED:
            filtered_df = df[df['event_num_in_case'] == self.prefix_length]
        else:
            filtered_df = df

        # Drop the helper column
        filtered_df = filtered_df.drop(columns=['event_num_in_case'])

        return filtered_df


    def _postprocess_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common postprocessing logic shared by all encoders. The method restores original ordering and drops unnecessary data.
        """
        # Restore original ordering
        df = df.sort_values(by=self.ORIGINAL_INDEX_KEY).reset_index(drop=True)

        # Drop unnecessary data
        df = df.drop(columns=[self.timestamp_key, self.ORIGINAL_INDEX_KEY])
        df = df.dropna(subset=[self.LABEL_KEY]).reset_index(drop=True)

        return df

    
    def _include_latest_payload(
        self,
        df: pd.DataFrame,
        attributes: str | list = 'all',
        categorical_attributes_encoding: CategoricalEncoding = CategoricalEncoding.STRING,
    ) -> pd.DataFrame:
        """
        Add latest payload attributes to encoded DataFrame. 
        """
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

        # Add latest payload of specified attributes to the dataframe
        for payload_attribute in attributes:
            attribute_values = []
            
            for _, row in df.iterrows():
                attribute_values.append(self.original_df.loc[row[self.ORIGINAL_INDEX_KEY], payload_attribute])

            df[f'{payload_attribute}_{self.LATEST_PAYLOAD_COL_SUFFIX_NAME}'] = attribute_values

        # Transform to one-hot if requested
        if categorical_attributes_encoding == CategoricalEncoding.ONE_HOT:
            categorical_columns = []
            
            for attribute in attributes:
                if is_object_dtype(df[f'{attribute}_{self.LATEST_PAYLOAD_COL_SUFFIX_NAME}']):
                    categorical_columns.append(f'{attribute}_{self.LATEST_PAYLOAD_COL_SUFFIX_NAME}')

            df = pd.get_dummies(
                df,
                columns=categorical_columns,
                drop_first=True,
            )

        return df
