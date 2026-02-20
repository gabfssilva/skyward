"""Pandas DataFrame Roundtrip.

Demonstrates that pandas DataFrames serialize seamlessly through
cloudpickle — both as arguments and return values. Generates a
~20 MB synthetic dataset locally, ships it to remote workers for
processing, and gets transformed DataFrames back.

    ┌────────┐  cloudpickle   ┌────────┐  cloudpickle   ┌────────┐
    │ Local  │ ──── 20 MB ──▶ │ Worker │ ──── result ──▶ │ Local  │
    │  DataFrame              │  pandas ops              │  DataFrame
    └────────┘                └────────┘                └────────┘
"""

import pandas as pd

import skyward as sky


@sky.compute
def hello() -> str:
    return "hi!"

@sky.compute
def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics on the remote worker."""
    return df.describe()


@sky.compute
def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue per product category."""
    import pandas as pd

    return (
        df.groupby("category")
        .agg(
            total_revenue=pd.NamedAgg("revenue", "sum"),
            avg_quantity=pd.NamedAgg("quantity", "mean"),
            order_count=pd.NamedAgg("revenue", "count"),
            avg_revenue=pd.NamedAgg("revenue", "mean"),
        )
        .sort_values("total_revenue", ascending=False)  # type: ignore[call-overload]
    )


@sky.compute
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering on the remote side — returns a new DataFrame."""
    df = df.copy()
    df["log_revenue"] = df["revenue"].apply("log1p")
    df["qty_bin"] = df["quantity"].pipe(
        lambda s: s.where(s <= 2, "3+").where(s > 2, s.astype(str))
    )
    df["revenue_rank"] = df.groupby("category")["revenue"].rank(pct=True)
    return df


def make_dataset(n_rows: int = 500000) -> pd.DataFrame:
    """Generate a synthetic e-commerce dataset (~20 MB)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)

    categories = ["Electronics", "Clothing", "Home", "Books", "Sports", "Toys"]
    regions = ["North", "South", "East", "West", "Central"]

    return pd.DataFrame({
        "order_id": np.arange(n_rows),
        "customer_id": rng.integers(1, 50_000, n_rows),
        "category": rng.choice(categories, n_rows),
        "region": rng.choice(regions, n_rows),
        "quantity": rng.integers(1, 10, n_rows),
        "revenue": rng.exponential(50, n_rows).round(2),
        "discount": rng.uniform(0, 0.3, n_rows).round(3),
        "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="s"),
    })


if __name__ == "__main__":
    import sys

    import pandas as pd

    df = make_dataset()
    size_mb = sys.getsizeof(df) / 1024 / 1024
    print(f"Dataset: {len(df):,} rows, {len(df.columns)} columns, ~{size_mb:.1f} MB")
    print(f"Columns: {list(df.columns)}\n")

    with sky.ComputePool(
        provider=sky.AWS(),
        nodes=1,
        image=sky.Image(pip=["pandas==2.3.3", "numpy"]),
        vcpus=2,
        memory_gb=2
    ) as pool:
        print(hello() >> pool)

        # Send full DataFrame, get summary back
        stats: pd.DataFrame = describe_dataset(df) >> pool
        print("Remote describe():")
        print(stats.to_string())

        # Aggregation on the remote side
        agg: pd.DataFrame = revenue_by_category(df) >> pool
        print("\nRevenue by category:")
        print(agg.to_string())

        # Feature engineering round-trip: send df, get enriched df back
        enriched: pd.DataFrame = add_features(df) >> pool
        print(f"\nEnriched DataFrame: {enriched.shape} — new columns: "
              f"{[c for c in enriched.columns if c not in df.columns]}")
        print(enriched.head(10).to_string())
