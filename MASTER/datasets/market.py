#%%
import numpy as np
import pandas as pd
#%%
def get_ratio(x, y):
    epsilon = 1e-7
    return x / np.abs(y + epsilon) - 1


def _validate_market_df(df: pd.DataFrame):
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_constituent_df(df: pd.DataFrame):
    required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def market_index_price(df, features, periods=[5, 10, 20, 30, 60], price_col="close", prefix="sp500"):
    """
    MASTER paper style market index price features.

    Features:
    - current market price at tau
    - historical mean/std of market price over past d' days

    Output feature count:
    1 + 2 * len(periods)
    """
    df[f"{prefix}_price_t"] = df[price_col]

    for period in periods:
        df[f"{prefix}_price_mean_{period}"] = df[price_col].rolling(window=period, min_periods=period).mean()
        df[f"{prefix}_price_std_{period}"] = df[price_col].rolling(window=period, min_periods=period).std(ddof=0)

    features.extend([f"{prefix}_price_t"])
    for period in periods:
        features.extend([
            f"{prefix}_price_mean_{period}",
            f"{prefix}_price_std_{period}",
        ])

    return df, features


def market_index_volume(df, features, periods=[5, 10, 20, 30, 60], volume_col="volume", prefix="sp500"):
    """
    MASTER paper style market index volume features.

    Features:
    - historical mean/std of market trading volume over past d' days

    Output feature count:
    2 * len(periods)
    """
    for period in periods:
        df[f"{prefix}_volume_mean_{period}"] = df[volume_col].rolling(window=period, min_periods=period).mean()
        df[f"{prefix}_volume_std_{period}"] = df[volume_col].rolling(window=period, min_periods=period).std(ddof=0)

    for period in periods:
        features.extend([
            f"{prefix}_volume_mean_{period}",
            f"{prefix}_volume_std_{period}",
        ])

    return df, features


def market_ohlcv_return_features(df, features, periods=[1, 5, 10, 20], prefix="sp500"):
    """
    Optional practical features from OHLCV.
    This part is not the core MASTER definition, but useful in practice.
    """
    for period in periods:
        df[f"{prefix}_open_return_{period}"] = df["open"].pct_change(periods=period)
        df[f"{prefix}_high_return_{period}"] = df["high"].pct_change(periods=period)
        df[f"{prefix}_low_return_{period}"] = df["low"].pct_change(periods=period)
        df[f"{prefix}_close_return_{period}"] = df["close"].pct_change(periods=period)
        df[f"{prefix}_volume_return_{period}"] = df["volume"].pct_change(periods=period)

        features.extend([
            f"{prefix}_open_return_{period}",
            f"{prefix}_high_return_{period}",
            f"{prefix}_low_return_{period}",
            f"{prefix}_close_return_{period}",
            f"{prefix}_volume_return_{period}",
        ])

    return df, features


def build_market_information_from_index(
    df,
    periods=[5, 10, 20, 30, 60],
    prefix="sp500",
    add_ohlcv_return_features=False,
    fillna=False,
):
    """
    Build market information directly from S&P500 index OHLCV data.

    Expected input columns:
    [date, open, high, low, close, volume]

    MASTER-style output:
    - current market price
    - price mean/std over periods
    - volume mean/std over periods

    If periods=[5,10,20,30,60], total core features = 21
    """
    df = df.copy()
    _validate_market_df(df)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features = []

    df, features = market_index_price(df, features, periods=periods, price_col="close", prefix=prefix)
    df, features = market_index_volume(df, features, periods=periods, volume_col="volume", prefix=prefix)

    if add_ohlcv_return_features:
        df, features = market_ohlcv_return_features(df, features, prefix=prefix)

    if fillna:
        df[features] = df[features].fillna(0.0)

    return df, features


def aggregate_constituents_to_market_index(
    df,
    weighted_by_market_cap=True,
    market_cap_col="market_cap",
    prefix="sp500"
):
    """
    Build a market-level OHLCV series from constituent-level S&P500 stock OHLCV.

    Expected input columns:
    [date, ticker, open, high, low, close, volume]
    optional: [market_cap]

    If weighted_by_market_cap=True:
        price fields are cap-weighted averages
    else:
        price fields are equally weighted averages

    volume is aggregated by sum.
    """
    df = df.copy()
    _validate_constituent_df(df)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    def weighted_avg(x, w):
        mask = x.notna() & w.notna()
        if mask.sum() == 0:
            return np.nan
        w_sum = w[mask].sum()
        if w_sum == 0:
            return np.nan
        return np.average(x[mask], weights=w[mask])

    grouped = df.groupby("date")

    market_df = pd.DataFrame(index=sorted(df["date"].unique()))
    market_df.index.name = "date"

    if weighted_by_market_cap:
        if market_cap_col not in df.columns:
            raise ValueError(f"'{market_cap_col}' column is required when weighted_by_market_cap=True")

        market_df["open"] = grouped.apply(lambda g: weighted_avg(g["open"], g[market_cap_col]))
        market_df["high"] = grouped.apply(lambda g: weighted_avg(g["high"], g[market_cap_col]))
        market_df["low"] = grouped.apply(lambda g: weighted_avg(g["low"], g[market_cap_col]))
        market_df["close"] = grouped.apply(lambda g: weighted_avg(g["close"], g[market_cap_col]))
    else:
        market_df["open"] = grouped["open"].mean()
        market_df["high"] = grouped["high"].mean()
        market_df["low"] = grouped["low"].mean()
        market_df["close"] = grouped["close"].mean()

    market_df["volume"] = grouped["volume"].sum()
    market_df["n_constituents"] = grouped["ticker"].nunique()

    market_df = market_df.reset_index()
    return market_df

#%%
def build_market_df_from_dict(data_dict):
    rows = []
    for ticker, df in data_dict.items():
        temp = df.copy()
        temp["ticker"] = ticker
        temp["date"] = temp.index
        rows.append(temp.reset_index(drop=True))
    return pd.concat(rows)

def build_market_information_from_constituents(
    df,
    periods=[5, 10, 20, 30, 60],
    weighted_by_market_cap=True,
    market_cap_col="market_cap",
    prefix="sp500",
    add_ohlcv_return_features=False,
    fillna=False,
):
    """
    Build MASTER-style market information from constituent OHLCV data.

    Step 1) Aggregate constituent data into market-level index OHLCV
    Step 2) Build market information features

    Input:
    [date, ticker, open, high, low, close, volume]
    optional: [market_cap]
    """
    df = build_market_df_from_dict(df)
    
    market_df = aggregate_constituents_to_market_index(
        df,
        weighted_by_market_cap=weighted_by_market_cap,
        market_cap_col=market_cap_col,
        prefix=prefix
    )

    market_info_df, features = build_market_information_from_index(
        market_df,
        periods=periods,
        prefix=prefix,
        add_ohlcv_return_features=add_ohlcv_return_features,
        fillna=fillna,
    )

    return market_info_df, features


def remove_outliers(df, features, threshold=1000):
    df = df.copy()
    for feat in features:
        df[feat] = df[feat].fillna(threshold)
        df.loc[df[feat] > threshold, feat] = threshold
        df.loc[df[feat] < -threshold, feat] = -threshold
    return df

