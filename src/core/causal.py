import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import nnls

def run_psm(df: pd.DataFrame, y: str, t: str, x: list[str]):
    # 1) Estimate propensity score
    lr = LogisticRegression(max_iter=1000)
    lr.fit(df[x], df[t])
    p = lr.predict_proba(df[x])[:,1]

    # 2) Nearest-neighbor matching on propensity score
    treated = df[df[t]==1].copy()
    control = df[df[t]==0].copy()
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(p[df[t]==0].reshape(-1,1))
    dist, idx = nn.kneighbors(p[df[t]==1].reshape(-1,1))
    matched_ctrl = control.iloc[idx.flatten()]

    te = treated[y].values - matched_ctrl[y].values
    ate = float(np.mean(te))
    return {"ATE_psm": ate, "n_pairs": int(len(te))}

def run_did(df: pd.DataFrame, y: str, t: str, time_col: str, pre_period: tuple, post_period: tuple):
    """
    Simple DID on aggregated means.
    pre_period/post_period: (start_inclusive, end_inclusive) strings parsable by pandas.to_datetime
    """
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    pre_mask  = (d[time_col] >= pd.to_datetime(pre_period[0])) & (d[time_col] <= pd.to_datetime(pre_period[1]))
    post_mask = (d[time_col] >= pd.to_datetime(post_period[0])) & (d[time_col] <= pd.to_datetime(post_period[1]))

    def _avg(mask, treat):
        sub = d[mask & (d[t]==treat)]
        return sub[y].mean()

    pre_t  = _avg(pre_mask, 1)
    pre_c  = _avg(pre_mask, 0)
    post_t = _avg(post_mask, 1)
    post_c = _avg(post_mask, 0)

    did = (post_t - pre_t) - (post_c - pre_c)
    return {
        "pre_treated_mean": float(pre_t),
        "pre_control_mean": float(pre_c),
        "post_treated_mean": float(post_t),
        "post_control_mean": float(post_c),
        "DID_effect": float(did)
    }

def run_synth_control(df: pd.DataFrame, y: str, unit_col: str, time_col: str,
                      treated_unit: str, treat_start: str):
    """
    Synthetic Control (toy NNLS version):
    - Build donor weights to match treated pre-period trajectory (non-negative, sum to 1).
    - Return post-period average treatment effect and per-period gaps.
    Assumes a single treated unit and multiple donor units.
    """
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    treat_start = pd.to_datetime(treat_start)

    # Pivot to unit x time
    piv = d.pivot_table(index=time_col, columns=unit_col, values=y, aggfunc='mean').sort_index()
    if treated_unit not in piv.columns:
        raise ValueError(f"Treated unit '{treated_unit}' not found in column '{unit_col}'.")

    donors = [c for c in piv.columns if c != treated_unit]
    if len(donors) == 0:
        raise ValueError("Need at least one donor unit.")

    pre = piv.index < treat_start

    Y1_pre = piv.loc[pre, treated_unit].values
    X0_pre = piv.loc[pre, donors].values

    # NNLS to get non-negative weights
    w, _ = nnls(X0_pre, Y1_pre)
    if w.sum() > 0:
        w = w / w.sum()

    # Construct synthetic for all times
    Y0_synth = piv[donors].values @ w
    Y1 = piv[treated_unit].values

    out = pd.DataFrame({
        "time": piv.index,
        "treated": Y1,
        "synthetic": Y0_synth,
        "gap": Y1 - Y0_synth
    })

    ate_post = out.loc[out["time"] >= treat_start, "gap"].mean()
    return {
        "weights": {donor: float(wi) for donor, wi in zip(donors, w)},
        "post_period_ATE": float(ate_post),
        "trajectory": out.to_dict(orient="list")
    }
