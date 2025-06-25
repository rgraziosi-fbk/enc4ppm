from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

class BaseEncoder(ABC):
    def __init__(self, case_id_key: str = 'case_id', activity_key: str = 'activity') -> None:
        """
        Initialize the BaseEncoder.

        Args:
            case_id_key: Column name for case identifiers.
            activity_key: Column name for activity names.
        """
        self.case_id_key = case_id_key
        self.activity_key = activity_key

    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the given event log DataFrame.

        Args:
            df: Event log with at least case_id, activity, and timestamp columns.

        Returns:
            Encoded feature representation.
        """
        pass
