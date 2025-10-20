from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

def _read(name: str) -> pd.DataFrame:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)

def load_core_data():
    """
    Load the key tables required for revenue aggregation.
    Returns: tuple of DataFrames
    """
    orders      = _read("orders.csv")
    order_items = _read("order_items.csv")
    payments    = _read("payments.csv")
    products    = _read("products.csv")
    return orders, order_items, payments, products

def build_monthly_revenue(orders, order_items, payments) -> pd.Series:
    """
    Build a monthly revenue time series:
    1. Prefer payments (cash-based view).
    2. Fallback to order_items totals if needed.
    """
    def to_date(s):
        s = pd.to_datetime(s, errors="coerce", utc=True)
        if s.dt.tz is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s

    if payments is not None and {"paid_at","amount"}.issubset(payments.columns):
        tmp = payments.copy()
        tmp["date"] = to_date(tmp["paid_at"]).dt.date
        tmp = tmp.dropna(subset=["date"])
        daily = tmp.groupby("date", as_index=False)["amount"].sum().rename(columns={"amount":"revenue"})
    else:
        oi = order_items.copy()
        od = orders.copy()
        od["date"] = to_date(od["order_date"]).dt.date
        df = oi.merge(od[["order_id","date"]], on="order_id", how="left").dropna(subset=["date"])
        daily = df.groupby("date", as_index=False)["item_total"].sum().rename(columns={"item_total":"revenue"})

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").set_index("date").asfreq("D", fill_value=0.0)
    monthly = daily["revenue"].resample("MS").sum().rename("revenue")
    monthly.index.name = "month"
    return monthly