from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

FIG = Path(__file__).resolve().parents[1] / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def _savefig(name):
    path = FIG / f"{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"[saved fig] {path}")

def plot_trend(monthly_rev):
    ax = monthly_rev.plot(title="Monthly Revenue")
    ax.set_ylabel("Revenue")
    _savefig("monthly_revenue_overview")
    plt.show()

def plot_growth_index(monthly_rev):
    base = monthly_rev.iloc[0] if len(monthly_rev) else 1.0
    growth_idx = (monthly_rev / base * 100)
    ax = growth_idx.plot(title="Revenue Growth Index (Base = 100)")
    ax.set_ylabel("Index")
    _savefig("revenue_growth_index")
    plt.show()

def plot_stl_decomposition(monthly_rev):
    stl = STL(monthly_rev, period=12, robust=True)
    res = stl.fit()
    fig = res.plot()
    fig.suptitle("STL Decomposition — Monthly Revenue", y=1.02)
    _savefig("stl_decomposition_monthly")
    plt.show()

def plot_forecast_with_ci(train, test, forecast, ci=None, title="Forecast with 95% CI"):
    fig, ax = plt.subplots()
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test", lw=2)
    forecast.plot(ax=ax, label="Forecast")
    if ci is not None:
        ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2, label="95% CI")
    ax.set_title(title)
    ax.set_ylabel("Revenue")
    ax.legend()
    _savefig("forecast_with_ci")
    plt.show()