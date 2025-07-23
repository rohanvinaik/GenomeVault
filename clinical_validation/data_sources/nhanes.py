import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
from .base import DataSourceBase

class NHANESDataSource(DataSourceBase):
    """NHANES data - no registration required"""

    BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"

    def __init__(self, cycle: str = "2017-2018"):
        super().__init__()
        self.cycle = cycle

    def load_data(self) -> pd.DataFrame:
        """Load NHANES data"""
        try:
            # Try to load key datasets
            glucose_url = f"{self.BASE_URL}/{self.cycle}/GLU_J.XPT"
            self._data = pd.read_sas(glucose_url)

            # Try to add HbA1c
            try:
                hba1c_url = f"{self.BASE_URL}/{self.cycle}/GHB_J.XPT"
                hba1c_df = pd.read_sas(hba1c_url)
                self._data = self._data.merge(hba1c_df, on='SEQN', how='left')
            except:
                self.logger.warning("Could not load HbA1c data")

            # Try to add diabetes diagnosis
            try:
                diabetes_url = f"{self.BASE_URL}/{self.cycle}/DIQ_J.XPT"
                diabetes_df = pd.read_sas(diabetes_url)
                self._data = self._data.merge(diabetes_df, on='SEQN', how='left')
            except:
                self.logger.warning("Could not load diabetes questionnaire")

            self._metadata['loaded_at'] = datetime.now()
            self._metadata['n_records'] = len(self._data)

            return self._data

        except Exception as e:
            self.logger.error(f"Failed to load NHANES: {e}")
            self._data = pd.DataFrame()
            return self._data

    def get_glucose_column(self) -> Optional[str]:
        return 'LBXGLU' if 'LBXGLU' in self._data.columns else None

    def get_hba1c_column(self) -> Optional[str]:
        return 'LBXGH' if 'LBXGH' in self._data.columns else None

    def get_outcome_column(self) -> Optional[str]:
        return 'DIQ010' if 'DIQ010' in self._data.columns else None

    def clean_data(self) -> pd.DataFrame:
        """Clean NHANES data"""
        df = self._data.copy()

        # Clean glucose
        if 'LBXGLU' in df.columns:
            df.loc[(df['LBXGLU'] < 20) | (df['LBXGLU'] > 600), 'LBXGLU'] = np.nan

        # Clean HbA1c
        if 'LBXGH' in df.columns:
            df.loc[(df['LBXGH'] < 3) | (df['LBXGH'] > 20), 'LBXGH'] = np.nan

        return df
