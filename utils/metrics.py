import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

# Troubleshooting for PEMS Nov 8
# def MAPE(pred, true):
#     return np.mean(np.abs((pred - true) / true))
def MAPE(pred, true):
    mape = np.abs((pred - true) / true)
    mape = np.where(mape > 5, 0, mape)
    return np.mean(mape)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    # Additional metrics
    # R2 (coefficient of determination)
    # Flatten to compute global score across all dims
    pred_flat = np.asarray(pred).reshape(-1)
    true_flat = np.asarray(true).reshape(-1)
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # Explained variance score
    var_diff = np.var(true_flat - pred_flat)
    var_true = np.var(true_flat)
    evs = 1 - var_diff / var_true if var_true != 0 else 0.0

    # Median absolute error
    medae = np.median(np.abs(pred_flat - true_flat))

    # Max error
    maxerr = np.max(np.abs(pred_flat - true_flat))

    # Symmetric MAPE (SMAPE)
    denom = np.abs(true_flat) + np.abs(pred_flat)
    # avoid div0
    smape_vals = np.zeros_like(denom)
    nonzero = denom != 0
    smape_vals[nonzero] = 2.0 * np.abs(pred_flat[nonzero] - true_flat[nonzero]) / denom[nonzero]
    smape = np.mean(smape_vals)

    return mae, mse, rmse, mape, mspe, r2, evs, medae, maxerr, smape
