import numpy as np
import pandas as pd
from scipy import stats

def ab_summary(df: pd.DataFrame, metric: str, denom: str, group_col: str = "treatment"):
    agg = df.groupby(group_col).agg({metric:"sum", denom:"sum"})
    agg["rate"] = agg[metric] / np.maximum(agg[denom], 1e-9)
    # two-proportion z-test
    c1, n1 = agg.loc[1, metric], agg.loc[1, denom]
    c0, n0 = agg.loc[0, metric], agg.loc[0, denom]
    p1, p0 = c1/n1, c0/n0
    p = (c1 + c0) / (n1 + n0)
    z = (p1 - p0) / np.sqrt(p*(1-p)*(1/n1 + 1/n0) + 1e-9)
    pval = 2*(1 - stats.norm.cdf(abs(z)))
    lift = (p1 - p0) / (p0 + 1e-9)
    return {
        "rate_treatment": float(p1),
        "rate_control": float(p0),
        "lift": float(lift),
        "z": float(z),
        "p_value": float(pval)
    }
