from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
import pandas as pd
import logging
from datetime import datetime


class DataSourceBase(ABC):
    """Base class for clinical data sources"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._data = None
        self._metadata = {
            "source_name": self.__class__.__name__,
            "loaded_at": None,
            "n_records": 0,
        }

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from source"""
        pass

    @abstractmethod
    def get_glucose_column(self) -> Optional[str]:
        """Return glucose column name"""
        pass

    @abstractmethod
    def get_hba1c_column(self) -> Optional[str]:
        """Return HbA1c column name"""
        pass

    @abstractmethod
    def get_outcome_column(self) -> Optional[str]:
        """Return outcome column name"""
        pass

    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        if self._data is None:
            return {}

        return {
            "n_records": len(self._data),
            "columns": list(self._data.columns),
            "missing_percentages": {
                col: (self._data[col].isna().sum() / len(self._data) * 100)
                for col in self._data.columns
            },
        }
