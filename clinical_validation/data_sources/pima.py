import pandas as pd
import numpy as np
from typing import Optional
from .base import DataSourceBase


class PimaDataSource(DataSourceBase):
    """Pima Indians Diabetes dataset"""

    URL = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    )

    def load_data(self) -> pd.DataFrame:
        """Load Pima dataset"""
        try:
            names = [
                "pregnancies",
                "glucose",
                "bp",
                "skin",
                "insulin",
                "bmi",
                "dpf",
                "age",
                "outcome",
            ]
            self._data = pd.read_csv(self.URL, names=names)

            # Replace zeros with NaN for impossible values
            for col in ["glucose", "bp", "bmi"]:
                self._data.loc[self._data[col] == 0, col] = np.nan

            self._metadata["loaded_at"] = datetime.now()
            self._metadata["n_records"] = len(self._data)

            return self._data

        except Exception as e:
            self.logger.error(f"Failed to load Pima data: {e}")
            self._data = pd.DataFrame()
            return self._data

    def get_glucose_column(self) -> Optional[str]:
        return "glucose"

    def get_hba1c_column(self) -> Optional[str]:
        return None  # Not available in Pima

    def get_outcome_column(self) -> Optional[str]:
        return "outcome"
