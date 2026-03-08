from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("pandas")

if TYPE_CHECKING:
    import pandas as pd

import pandas as pd

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120), pytest.mark.xdist_group("pandas_pool")]


def _titanic() -> pd.DataFrame:
    return pd.read_csv(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )


class TestPandasDataFrame:
    def test_titanic_imputation(self, pandas_pool):
        df = _titanic()
        nulls_before = int(df.isnull().sum().sum())

        @sky.function
        def impute(df: pd.DataFrame) -> pd.DataFrame:
            for col in df.select_dtypes(include="number"):
                df[col] = df[col].fillna(float(df[col].median()))
            for col in df.select_dtypes(include="object"):
                df[col] = df[col].fillna(str(df[col].mode()[0]))
            return df

        result = impute(df) >> pandas_pool

        assert isinstance(result, pd.DataFrame)
        assert nulls_before > 0
        assert result.isnull().sum().sum() == 0
        assert len(result) == 891
