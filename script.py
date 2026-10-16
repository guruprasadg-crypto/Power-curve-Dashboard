import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 1. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Jagaluru WTG Power Curve Analysis",
    page_icon="🌬️",
    layout="wide",
)

st.title("🌬️ Wind Turbine Power Curve Analysis")
st.caption(
    "Jagaluru Site | EN-156 3.3 MW | "
    "10-minute SCADA statistics vs reference power curve"
)


# =========================================================
# 2. CONFIGURATION
# =========================================================

WTG_COL = "Name"
TIME_COL = "Time"
STATUS_COL = "PCurveStsAve"
WIND_SPEED_COL = "WindSpeedAve"
WIND_STD_COL = "WindSpeedStd"
PITCH_STD_COL = "BladePitchStd"
POWER_COL = "ActivePWAve"

DEFAULT_BIN_WIDTH = 0.5


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def read_scada(uploaded_file):
    """Read SCADA CSV with common encoding fallbacks."""
    errors = []

    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                encoding=encoding,
                low_memory=False,
            )
            df.columns = df.columns.astype(str).str.strip()
            return df
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError(
        "Could not read the SCADA CSV. "
        + " | ".join(errors)
    )


def read_reference(uploaded_file, sheet_name=None):
    """Read reference curve from Excel or CSV."""
    filename = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if filename.endswith((".xlsx", ".xls")):
        if sheet_name is None:
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_excel(
                uploaded_file,
                sheet_name=sheet_name,
            )
    else:
        df = pd.read_csv(uploaded_file)

    df.columns = df.columns.astype(str).str.strip()

    # The first two columns are assumed to be:
    # Reference wind speed and reference power.
    if df.shape[1] < 2:
        raise ValueError(
            "The reference file must contain at least two columns."
        )

    ref = df.iloc[:, :2].copy()
    ref.columns = [
        "ReferenceWindSpeed_mps",
        "ReferencePower_kW",
    ]

    for col in ref.columns:
        ref[col] = pd.to_numeric(ref[col], errors="coerce")

    ref = (
        ref.dropna()
        .sort_values("ReferenceWindSpeed_mps")
        .drop_duplicates(
            subset="ReferenceWindSpeed_mps",
            keep="mean",
        )
    )

    if len(ref) < 2:
        raise ValueError(
            "The reference curve needs at least two valid points."
        )

    return ref


def analyze_data(
    scada,
    ref,
    bin_width,
    status_value,
    wind_std_limit,
    pitch_min,
    pitch_max,
):
    """Apply filters and calculate power curve metrics."""

    required_cols = [
        WTG_COL,
        TIME_COL,
        STATUS_COL,
        WIND_SPEED_COL,
        WIND_STD_COL,
        PITCH_STD_COL,
        POWER_COL,
    ]

    missing = [
        col for col in required_cols
        if col not in scada.columns
    ]

    if missing:
        raise ValueError(
            f"Missing SCADA columns: {missing}\n"
            f"Available columns: {scada.columns.tolist()}"
        )

    df = scada.copy()

    for col in [
        STATUS_COL,
        WIND_SPEED_COL,
        WIND_STD_COL,
        PITCH_STD_COL,
        POWER_COL,
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL], errors="coerce"
    )

    # Required filtration conditions
    mask_status = df[STATUS_COL].eq(status_value)
    mask_wind_std = df[WIND_STD_COL].lt(wind_std_limit)
    mask_pitch_std = df[PITCH_STD_COL].between(
        pitch_min,
        pitch_max,
        inclusive="both",
    )

    # Also exclude records missing required analysis values.
    mask_valid = df[
        [WTG_COL, WIND_SPEED_COL, POWER_COL]
    ].notna().all(axis=1)

    combined_mask = (
        mask_status
        & mask_wind_std
        & mask_pitch_std
        & mask_valid
    )

    filtered = df.loc[combined_mask].copy()

    filter_counts = pd.DataFrame({
        "Filter Stage": [
            "Total SCADA records",
            f"{STATUS_COL} = {status_value}",
            f"{WIND_STD_COL} < {wind_std_limit}",
            f"{PITCH_STD_COL} between {pitch_min} and {pitch_max}",
            "All filters + valid WTG, wind speed and power",
        ],
        "Record Count": [
            len(df),
            int(mask_status.sum()),
            int((mask_status & mask_wind_std).sum()),
            int(
                (
                    mask_status
                    & mask_wind_std
                    & mask_pitch_std
                ).sum()
            ),
            int(combined_mask.sum()),
        ],
    })

    filtered = filtered.rename(columns={
        WTG_COL: "WTG",
        TIME_COL: "DateTime",
        WIND_SPEED_COL: "WindSpeed_mps",
        POWER_COL: "ActualPower_kW",
    })

    # Match SCADA wind speeds to the reference curve.
    ref_min = ref["ReferenceWindSpeed_mps"].min()
    ref_max = ref["ReferenceWindSpeed_mps"].max()

    # Do not extrapolate outside the reference curve range.
    filtered = filtered[
        filtered["WindSpeed_mps"].between(ref_min, ref_max)
    ].copy()

    if filtered.empty:
        return (
            filtered,
            filter_counts,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            ref_min,
            ref_max,
        )

    filtered["ReferencePower_kW"] = np.interp(
        filtered["WindSpeed_mps"],
        ref["ReferenceWindSpeed_mps"],
        ref["ReferencePower_kW"],
    )

    # Positive deviation = actual power above reference.
    # Negative deviation = actual power below reference.
    filtered["Deviation_kW"] = (
        filtered["ActualPower_kW"]
        - filtered["ReferencePower_kW"]
    )

    filtered["Shortfall_kW"] = (
        filtered["ReferencePower_kW"]
        - filtered["ActualPower_kW"]
    )

    filtered["Deviation_pct"] = np.where(
        filtered["ReferencePower_kW"] > 0,
        100 * filtered["Deviation_kW"]
        / filtered["ReferencePower_kW"],
        np.nan,
    )

    filtered["WindSpeedBin_mps"] = (
        np.floor(
            filtered["WindSpeed_mps"] / bin_width
        ) * bin_width
    )

    # Fleet summary by wind-speed bin.
    fleet_bins = (
        filtered.groupby("WindSpeedBin_mps")
        .agg(
            Samples=("ActualPower_kW", "size"),
            MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
            MeanActualPower_kW=("ActualPower_kW", "mean"),
            MeanReferencePower_kW=("ReferencePower_kW", "mean"),
            MeanDeviation_kW=("Deviation_kW", "mean"),
            MeanShortfall_kW=("Shortfall_kW", "mean"),
            MeanDeviation_pct=("Deviation_pct", "mean"),
        )
        .reset_index()
        .sort_values("WindSpeedBin_mps")
    )

    # Individual WTG summary.
    wtg_summary = (
        filtered.groupby("WTG")
        .agg(
            ValidSamples=("ActualPower_kW", "size"),
            MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
            MeanActualPower_kW=("ActualPower_kW", "mean"),
            MeanReferencePower_kW=("ReferencePower_kW", "mean"),
            MeanDeviation_kW=("Deviation_kW", "mean"),
            MeanSampleDeviation_pct=("Deviation_pct", "mean"),
            FirstTimestamp=("DateTime", "min"),
            LastTimestamp=("DateTime", "max"),
        )
        .reset_index()
    )

    wtg_summary["AggregateDeviation_pct"] = np.where(
        wtg_summary["MeanReferencePower_kW"] > 0,
        100 * (
            wtg_summary["MeanActualPower_kW"]
            - wtg_summary["MeanReferencePower_kW"]
        ) / wtg_summary["MeanReferencePower_kW"],
        np.nan,
    )

    wtg_summary = wtg_summary.sort_values(
        "AggregateDeviation_pct",
        na_position="last",
    )

    # WTG-specific wind-speed bins.
    wtg_bins = (
        filtered.groupby(["WTG", "WindSpeedBin_mps"])
        .agg(
            Samples=("ActualPower_kW", "size"),
            MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
            MeanActualPower_kW=("ActualPower_kW", "mean"),
            MeanReferencePower_kW=("ReferencePower_kW", "mean"),
            MeanDeviation_kW=("Deviation_kW", "mean"),
            MeanShortfall_kW=("Shortfall_kW", "mean"),
            MeanDeviation_pct=("Deviation_pct", "mean"),
        )
        .reset_index()
    )

    return (
        filtered,
        filter_counts,
        fleet_bins,
        wtg_summary,
        wtg_bins,
        ref_min,
        ref_max,
    )


def make_excel_report(
    filtered,
    ref,
    filter_counts,
    fleet_bins,
    wtg_summary,
    wtg_bins,
):
    """Build a downloadable multi-sheet Excel report."""

    output = io.BytesIO()

    details_cols = [
        "WTG",
        "DateTime",
        STATUS_COL,
        WIND_STD_COL,
        PITCH_STD_COL,
        "WindSpeed_mps",
        "ActualPower_kW",
        "ReferencePower_kW",
        "Deviation_kW",
        "Shortfall_kW",
        "Deviation_pct",
        "WindSpeedBin_mps",
    ]

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        filter_counts.to_excel(
            writer, sheet_name="Filter Counts", index=False
        )
        wtg_summary.to_excel(
            writer, sheet_name="WTG Summary", index=False
        )
        fleet_bins.to_excel(
            writer, sheet_name="Fleet Wind Bins", index=False
        )
        wtg_bins.to_excel(
            writer, sheet_name="WTG Wind Bins", index=False
        )
        filtered[details_cols].to_excel(
            writer,
            sheet_name="Filtered SCADA Details",
            index=False,
        )
        ref.to_excel(
            writer, sheet_name="Reference Curve", index=False
        )

    output.seek(0)
    return output.getvalue()


# =========================================================
# 4. SIDEBAR: INPUT FILES AND FILTERS
# =========================================================

st.sidebar.header("Input Files")

scada_file = st.sidebar.file_uploader(
    "Upload 10-minute SCADA statistics CSV",
    type=["csv"],
    key="scada_upload",
)

reference_file = st.sidebar.file_uploader(
    "Upload reference power curve Excel/CSV",
    type=["xlsx", "xls", "csv"],
    key="reference_upload",
)

st.sidebar.header("Filtration Conditions")

status_value = st.sidebar.number_input(
    "PCurveStsAve equals",
    value=100,
    step=1,
)

wind_std_limit = st.sidebar.number_input(
    "WindSpeedStd less than",
    min_value=0.01,
    value=1.0,
    step=0.1,
)

pitch_min = st.sidebar.number_input(
    "BladePitchStd minimum",
    value=-5.0,
    step=1.0,
)

pitch_max = st.sidebar.number_input(
    "BladePitchStd maximum",
    value=19.0,
    step=1.0,
)

bin_width = st.sidebar.selectbox(
    "Wind-speed bin width (m/s)",
    options=[0.5, 1.0, 0.25],
    index=0,
)

st.sidebar.caption(
    "Pitch limits are inclusive. WindSpeedStd uses a strict less-than condition."
)


# =========================================================
# 5. PROCESS FILES
# =========================================================

if scada_file is None or reference_file is None:
    st.info(
        "Upload both files in the sidebar to start the analysis."
    )

    st.markdown("""
    **Expected SCADA columns**

    - A: `Name` — WTG identifier
    - B: `Time` — timestamp
    - C: `PCurveStsAve` — status
    - G: `WindSpeedAve` — wind speed (m/s)
    - J: `WindSpeedStd` — wind-speed standard deviation
    - R: `BladePitchStd` — blade-pitch standard deviation
    - S: `ActivePWAve` — active power (kW)

    The reference file's first two columns must contain
    reference wind speed (m/s) and reference power (kW).
    """)
    st.stop()


try:
    scada = read_scada(scada_file)
    ref = read_reference(reference_file)

    (
        filtered,
        filter_counts,
        fleet_bins,
        wtg_summary,
        wtg_bins,
        ref_min,
        ref_max,
    ) = analyze_data(
        scada=scada,
        ref=ref,
        bin_width=bin_width,
        status_value=status_value,
        wind_std_limit=wind_std_limit,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
    )

except Exception as exc:
    st.error(f"Could not process the files: {exc}")
    st.stop()


if filtered.empty:
    st.warning(
        "No records matched all filters and the reference curve range. "
        "Check the input columns, units, filters, and reference curve."
    )
    st.dataframe(filter_counts, use_container_width=True)
    st.stop()


# =========================================================
# 6. KPI CARDS
# =========================================================

total_records = len(scada)
filtered_records = int(filter_counts.iloc[-1]["Record Count"])
matched_records = len(filtered)
wtg_count = filtered["WTG"].nunique()

mean_actual = filtered["ActualPower_kW"].mean()
mean_reference = filtered["ReferencePower_kW"].mean()

overall_deviation_pct = (
    100 * (mean_actual - mean_reference) / mean_reference
    if mean_reference > 0 else np.nan
)

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total SCADA Records", f"{total_records:,}")
k2.metric("Passing Filters", f"{filtered_records:,}")
k3.metric("Matched to Reference", f"{matched_records:,}")
k4.metric("WTGs Analysed", f"{wtg_count:,}")
k5.metric(
    "Mean Power Deviation",
    f"{overall_deviation_pct:.2f}%"
    if pd.notna(overall_deviation_pct) else "N/A",
)

st.caption(
    f"Reference curve range: {ref_min:.2f}–{ref_max:.2f} m/s. "
    "Mean power deviation is actual mean power versus mean interpolated "
    "reference power; it is not a direct energy-loss estimate."
)


# =========================================================
# 7. FILTER COUNTS
# =========================================================

with st.expander("View filtration counts", expanded=False):
    st.dataframe(
        filter_counts,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# 8. INTERACTIVE POWER CURVE GRAPHS
# =========================================================

st.subheader("Power Curve Comparison")

wtg_options = ["All WTGs"] + sorted(
    filtered["WTG"].astype(str).unique().tolist()
)

selected_wtg = st.selectbox(
    "Select WTG",
    options=wtg_options,
)

if selected_wtg == "All WTGs":
    chart_data = filtered
    bin_data = fleet_bins
    chart_title = "Jagaluru Fleet Power Curve"
else:
    chart_data = filtered[
        filtered["WTG"].astype(str) == selected_wtg
    ]
    bin_data = (
        chart_data.groupby("WindSpeedBin_mps")
        .agg(
            Samples=("ActualPower_kW", "size"),
            MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
            MeanActualPower_kW=("ActualPower_kW", "mean"),
            MeanReferencePower_kW=("ReferencePower_kW", "mean"),
            MeanDeviation_kW=("Deviation_kW", "mean"),
            MeanShortfall_kW=("Shortfall_kW", "mean"),
            MeanDeviation_pct=("Deviation_pct", "mean"),
        )
        .reset_index()
        .sort_values("WindSpeedBin_mps")
    )
    chart_title = f"{selected_wtg} Power Curve"

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ref["ReferenceWindSpeed_mps"],
    y=ref["ReferencePower_kW"],
    mode="lines+markers",
    name="Reference Power Curve",
    line=dict(width=3, color="black"),
))

fig.add_trace(go.Scattergl(
    x=chart_data["WindSpeed_mps"],
    y=chart_data["ActualPower_kW"],
    mode="markers",
    name="Filtered SCADA (10 min)",
    marker=dict(size=5, opacity=0.25),
    customdata=chart_data[["WTG", "DateTime"]].astype(str).to_numpy(),
    hovertemplate=(
        "Wind speed: %{x:.2f} m/s<br>"
        "Actual power: %{y:.1f} kW<br>"
        "WTG: %{customdata[0]}<br>"
        "Time: %{customdata[1]}<extra></extra>"
    ),
))

fig.add_trace(go.Scatter(
    x=bin_data["MeanWindSpeed_mps"],
    y=bin_data["MeanActualPower_kW"],
    mode="lines+markers",
    name=f"SCADA mean ({bin_width} m/s bins)",
    line=dict(width=2.5),
    marker=dict(size=8),
))

fig.update_layout(
    title=chart_title,
    xaxis_title="Wind Speed (m/s)",
    yaxis_title="Active Power (kW)",
    template="plotly_white",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True, rangemode="tozero")

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 9. POWER DEVIATION GRAPH
# =========================================================

st.subheader("Power Deviation by Wind-Speed Bin")

dev_fig = go.Figure()

dev_fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="black",
)

dev_fig.add_trace(go.Bar(
    x=bin_data["MeanWindSpeed_mps"],
    y=bin_data["MeanDeviation_kW"],
    name="Mean deviation",
    customdata=bin_data[["Samples"]].to_numpy(),
    hovertemplate=(
        "Mean wind speed: %{x:.2f} m/s<br>"
        "Actual - reference: %{y:.1f} kW<br>"
        "Samples: %{customdata[0]}<extra></extra>"
    ),
))

dev_fig.update_layout(
    title=f"{selected_wtg} - Actual Power minus Reference Power",
    xaxis_title="Mean Wind Speed (m/s)",
    yaxis_title="Mean Deviation (kW)",
    template="plotly_white",
)

st.plotly_chart(dev_fig, use_container_width=True)

st.caption(
    "Negative deviation means actual power is below the reference curve. "
    "Positive deviation means actual power is above the reference curve."
)


# =========================================================
# 10. WTG PERFORMANCE SUMMARY
# =========================================================

st.subheader("WTG Performance Summary")

st.caption(
    "WTGs are sorted by aggregate power deviation. "
    "Negative values indicate mean actual power below mean reference power."
)

st.dataframe(
    wtg_summary,
    hide_index=True,
    use_container_width=True,
)


# =========================================================
# 11. WIND-SPEED BIN SUMMARY
# =========================================================

st.subheader("Wind-Speed Bin Analysis")

st.dataframe(
    bin_data,
    hide_index=True,
    use_container_width=True,
)

with st.expander("View filtered SCADA records"):
    st.dataframe(
        chart_data[
            [
                "WTG",
                "DateTime",
                STATUS_COL,
                WIND_STD_COL,
                PITCH_STD_COL,
                "WindSpeed_mps",
                "ActualPower_kW",
                "ReferencePower_kW",
                "Deviation_kW",
                "Shortfall_kW",
                "Deviation_pct",
                "WindSpeedBin_mps",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# 12. EXCEL DOWNLOAD
# =========================================================

excel_bytes = make_excel_report(
    filtered=filtered,
    ref=ref,
    filter_counts=filter_counts,
    fleet_bins=fleet_bins,
    wtg_summary=wtg_summary,
    wtg_bins=wtg_bins,
)

st.subheader("Download Analysis Report")

st.download_button(
    label="📥 Download Power Curve Analysis Excel",
    data=excel_bytes,
    file_name="Jagaluru_Power_Curve_Analysis.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
)

st.success("Analysis completed successfully.")
