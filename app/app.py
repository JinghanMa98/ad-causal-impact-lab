
import streamlit as st
import pandas as pd
from pathlib import Path
from src.core.abtest import ab_summary
from src.core.causal import run_psm, run_did, run_synth_control

st.title("Ads Causal & A/B Toolkit (Demo)")

default_path = Path("data/ads_sample.csv")
file = st.file_uploader("Upload CSV (optional)", type=["csv"])
if file:
    df = pd.read_csv(file)
else:
    df = pd.read_csv(default_path)
st.caption(f"Rows: {len(df)}")
st.dataframe(df.head())

st.header("Choose Method")
method = st.selectbox("Method", ["A/B (two-proportion)", "PSM (ATE)", "Difference-in-Differences", "Synthetic Control"])

if method == "A/B (two-proportion)":
    metric = st.selectbox("Numerator (metric)", options=[c for c in df.columns if c not in ["timestamp"]], index=4 if "clicks" in df.columns else 0)
    denom = st.selectbox("Denominator", options=[c for c in df.columns if c not in ["timestamp"]], index=3 if "impressions" in df.columns else 0)
    group_col = st.selectbox("Group column (0/1)", options=df.columns, index=list(df.columns).index("treatment") if "treatment" in df.columns else 0)
    if st.button("Run A/B"):
        res = ab_summary(df, metric=metric, denom=denom, group_col=group_col)
        st.json(res)

elif method == "PSM (ATE)":
    y = st.selectbox("Outcome (y)", options=df.columns, index=list(df.columns).index("conversions") if "conversions" in df.columns else 0)
    t = st.selectbox("Treatment (t)", options=df.columns, index=list(df.columns).index("treatment") if "treatment" in df.columns else 0)
    feature_cols = st.multiselect("Covariates (X)", options=[c for c in df.columns if c not in [y, t]], default=[c for c in ["impressions","clicks","spend"] if c in df.columns])
    if st.button("Run PSM"):
        res = run_psm(df, y=y, t=t, x=feature_cols)
        st.json(res)

elif method == "Difference-in-Differences":
    y = st.selectbox("Outcome (y)", options=df.columns, index=list(df.columns).index("conversions") if "conversions" in df.columns else 0)
    t = st.selectbox("Treatment (t)", options=df.columns, index=list(df.columns).index("treatment") if "treatment" in df.columns else 0)
    time_col = st.selectbox("Time column", options=df.columns, index=list(df.columns).index("timestamp") if "timestamp" in df.columns else 0)
    pre_start = st.text_input("Pre-period start (YYYY-MM-DD)", "2025-07-01")
    pre_end   = st.text_input("Pre-period end (YYYY-MM-DD)", "2025-07-01")
    post_start= st.text_input("Post-period start (YYYY-MM-DD)", "2025-07-02")
    post_end  = st.text_input("Post-period end (YYYY-MM-DD)", "2025-07-10")
    if st.button("Run DiD"):
        res = run_did(df, y=y, t=t, time_col=time_col, pre_period=(pre_start, pre_end), post_period=(post_start, post_end))
        st.json(res)

elif method == "Synthetic Control":
    y = st.selectbox("Outcome (y)", options=df.columns, index=list(df.columns).index("conversions") if "conversions" in df.columns else 0)
    unit_col = st.selectbox("Unit column (treated + donors)", options=df.columns, index=list(df.columns).index("region") if "region" in df.columns else 0)
    time_col = st.selectbox("Time column", options=df.columns, index=list(df.columns).index("timestamp") if "timestamp" in df.columns else 0)
    treated_unit = st.text_input("Treated unit value", value="NE")
    treat_start = st.text_input("Treatment start (YYYY-MM-DD)", "2025-07-02")
    if st.button("Run Synthetic Control"):
        res = run_synth_control(df, y=y, unit_col=unit_col, time_col=time_col, treated_unit=treated_unit, treat_start=treat_start)
        st.json({"weights": res["weights"], "post_period_ATE": res["post_period_ATE"]})
        out = pd.DataFrame(res["trajectory"])
        out["time"] = pd.to_datetime(out["time"])
        out = out.set_index("time")[["treated","synthetic"]]
        st.line_chart(out)
