import pandas as pd

def build_exogenous_features(marketing=None, seo=None, promos=None, orders=None) -> pd.DataFrame | None:
    """
    Create a monthly exogenous feature matrix from optional daily marketing,
    SEO and promotions data.
    """
    def monthly_sum(df, date_col, value_col, alias):
        d = df.copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col])
        d["month"] = d[date_col].dt.to_period("M").dt.to_timestamp()
        return d.groupby("month")[value_col].sum().rename(alias)

    parts = []
    if marketing is not None and {"date","spend"}.issubset(marketing.columns):
        parts.append(monthly_sum(marketing, "date", "spend", "mkt_spend"))
    if seo is not None and {"date","clicks"}.issubset(seo.columns):
        parts.append(monthly_sum(seo, "date", "clicks", "seo_clicks"))
    if promos is not None and {"order_id","promotion_id"}.issubset(promos.columns) and orders is not None:
        pa = promos.merge(orders[["order_id","order_date"]], on="order_id", how="left")
        parts.append(monthly_sum(pa, "order_date", "promotion_id", "promo_count"))

    if parts:
        exog = pd.concat(parts, axis=1).sort_index().fillna(0.0)
        return exog
    return None

def train_test_split_ts(series: pd.Series, cutoff: str):
    """
    Non-overlapping time-series split:
    train = months < cutoff
    test  = months >= cutoff
    """
    cut = pd.Timestamp(cutoff)
    return series.loc[series.index < cut], series.loc[series.index >= cut]