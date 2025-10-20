import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

def eval_and_store(name, y_true, y_pred, metrics_list):
    """
    Aligns and computes MAE, RMSE, MAPE, sMAPE.
    Appends a metrics dict to metrics_list.
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred).reindex(y_true.index)
    aligned = pd.concat([y_true, y_pred], axis=1, keys=["yt","yp"]).dropna()

    yt, yp = aligned["yt"].values.astype(float), aligned["yp"].values.astype(float)
    mae  = mean_absolute_error(yt, yp)
    rmse = mean_squared_error(yt, yp) ** 0.5
    mape = (np.abs((yt - yp) / np.where(yt == 0, 1e-9, yt))).mean() * 100
    s_mape = 100 * np.mean(2 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-9))
    metrics_list.append({"model": name, "MAE": mae, "RMSE": rmse, "MAPE%": mape, "sMAPE%": s_mape})

def fit_baselines(train, test, metrics_list):
    """Naive and 3-month Moving Average baselines."""
    naive_fc = pd.Series(train.iloc[-1], index=test.index)
    eval_and_store("Naive (last)", test, naive_fc, metrics_list)

    k = 3
    ma_fc = pd.Series(index=test.index, dtype=float)
    hist = train.copy()
    for m in test.index:
        ma_fc.loc[m] = hist.tail(k).mean()
        hist = pd.concat([hist, pd.Series([test.loc[m]], index=[m])])
    eval_and_store(f"MovingAvg(k={k})", test, ma_fc, metrics_list)
    return naive_fc, ma_fc

def fit_ets(train, test, metrics_list):
    """Triple Exponential Smoothing (additive)."""
    ets = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
    ets_fit = ets.fit(optimized=True)
    ets_fc = pd.Series(ets_fit.forecast(len(test)), index=test.index)
    eval_and_store("ETS(AAA)", test, ets_fc, metrics_list)
    return ets_fit, ets_fc

def fit_sarima(train, test, metrics_list):
    """Simple grid search SARIMA without exogenous."""
    p = d = q = range(0,2)
    pdq = list(itertools.product(p,d,q))
    seasonal_pdq = [(pi,di,qi,12) for (pi,di,qi) in pdq]
    best_aic, best_cfg, best_fit = np.inf, None, None
    for order in pdq:
        for sorder in seasonal_pdq:
            try:
                model = SARIMAX(train, order=order, seasonal_order=sorder,
                                enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best_aic, best_cfg, best_fit = res.aic, (order, sorder), res
            except:
                pass
    fc = pd.Series(best_fit.get_forecast(steps=len(test)).predicted_mean,
                   index=test.index)
    eval_and_store(f"SARIMA{best_cfg}", test, fc, metrics_list)
    return best_fit, fc, best_cfg

def fit_sarimax(train, test, train_exog, test_exog, metrics_list):
    """SARIMAX with exogenous regressors."""
    p = d = q = range(0,2)
    pdq = list(itertools.product(p,d,q))
    seasonal_pdq = [(pi,di,qi,12) for (pi,di,qi) in pdq]
    best_aic, best_cfg, best_fit = np.inf, None, None
    for order in pdq:
        for sorder in seasonal_pdq:
            try:
                model = SARIMAX(train, exog=train_exog,
                                order=order, seasonal_order=sorder,
                                enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best_aic, best_cfg, best_fit = res.aic, (order, sorder), res
            except:
                pass
    fc = None
    if best_fit is not None:
        fc = pd.Series(best_fit.get_forecast(steps=len(test), exog=test_exog).predicted_mean,
                       index=test.index)
        eval_and_store(f"SARIMAX{best_cfg}+exog", test, fc, metrics_list)
    return best_fit, fc, best_cfg