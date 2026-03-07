from __future__ import annotations

import pandas as pd
import pytest

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
                df[col] = df[col].fillna(df[col].median())
            for col in df.select_dtypes(include="object"):
                df[col] = df[col].fillna(df[col].mode()[0])
            return df

        result = impute(df) >> pandas_pool

        assert isinstance(result, pd.DataFrame)
        assert nulls_before > 0
        assert result.isnull().sum().sum() == 0
        assert len(result) == 891
