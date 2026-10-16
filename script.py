from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Keep this script and both input files in the same folder.
BASE_DIR = Path(__file__).resolve().parent
SCADA_FILE = BASE_DIR / "EN-156-3.3_20260901_20260930_ten_minutes.csv"
REFERENCE_FILE = BASE_DIR / "Refernce PC - Jagaluru.xlsx"
OUTPUT_DIR = BASE_DIR / "Jagaluru_Power_Curve_Output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Column names from the supplied files
WTG_COL = "Name"                 # A
TIME_COL = "Time"                # B
STATUS_COL = "PCurveStsAve"      # C
WIND_SPEED_COL = "WindSpeedAve"  # G
WIND_STD_COL = "WindSpeedStd"    # J
PITCH_STD_COL = "BladePitchStd"  # R
POWER_COL = "ActivePWAve"        # S
BIN_WIDTH = 0.5

scada = pd.read_csv(SCADA_FILE)
ref = pd.read_excel(REFERENCE_FILE).iloc[:, :2].copy()
ref.columns = ["ReferenceWindSpeed_mps", "ReferencePower_kW"]

for col in [STATUS_COL, WIND_STD_COL, PITCH_STD_COL,
            WIND_SPEED_COL, POWER_COL]:
    scada[col] = pd.to_numeric(scada[col], errors="coerce")
scada[TIME_COL] = pd.to_datetime(scada[TIME_COL], errors="coerce")
for col in ["ReferenceWindSpeed_mps", "ReferencePower_kW"]:
    ref[col] = pd.to_numeric(ref[col], errors="coerce")
ref = (ref.dropna().sort_values("ReferenceWindSpeed_mps")
       .drop_duplicates("ReferenceWindSpeed_mps"))

# Required filters
mask_status = scada[STATUS_COL].eq(100)
mask_wind_std = scada[WIND_STD_COL].lt(1)
mask_pitch_std = scada[PITCH_STD_COL].between(-5, 19, inclusive="both")
mask = mask_status & mask_wind_std & mask_pitch_std
filtered = scada.loc[mask].copy()

filter_counts = pd.DataFrame([
    ["Total SCADA records", len(scada)],
    ["PCurveStsAve = 100", int(mask_status.sum())],
    ["After PCurveStsAve = 100 and WindSpeedStd < 1",
     int((mask_status & mask_wind_std).sum())],
    ["After all three required filters", int(mask.sum())],
], columns=["Filter Stage", "Record Count"])

filtered = filtered.rename(columns={
    WTG_COL: "WTG",
    TIME_COL: "DateTime",
    WIND_SPEED_COL: "WindSpeed_mps",
    POWER_COL: "ActualPower_kW",
})

# Match reference power by linear interpolation. Do not extrapolate.
ref_min = ref["ReferenceWindSpeed_mps"].min()
ref_max = ref["ReferenceWindSpeed_mps"].max()
filtered = filtered[
    filtered["WindSpeed_mps"].between(ref_min, ref_max)
].copy()
filtered["ReferencePower_kW"] = np.interp(
    filtered["WindSpeed_mps"],
    ref["ReferenceWindSpeed_mps"],
    ref["ReferencePower_kW"]
)
filtered["Deviation_kW"] = (
    filtered["ActualPower_kW"] - filtered["ReferencePower_kW"]
)
filtered["Shortfall_kW"] = -filtered["Deviation_kW"]
filtered["Deviation_pct"] = np.where(
    filtered["ReferencePower_kW"] > 0,
    100 * filtered["Deviation_kW"] / filtered["ReferencePower_kW"],
    np.nan
)
filtered["WindSpeedBin_mps"] = (
    np.floor(filtered["WindSpeed_mps"] / BIN_WIDTH) * BIN_WIDTH
)

# Wind-speed-bin summary by WTG
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
        PowerStdDev_kW=("ActualPower_kW", "std"),
    ).reset_index()
)

# Fleet bin summary
fleet_bins = (
    filtered.groupby("WindSpeedBin_mps")
    .agg(
        Samples=("ActualPower_kW", "size"),
        MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
        MeanActualPower_kW=("ActualPower_kW", "mean"),
        MeanReferencePower_kW=("ReferencePower_kW", "mean"),
        MeanDeviation_kW=("Deviation_kW", "mean"),
        MeanDeviation_pct=("Deviation_pct", "mean"),
    ).reset_index()
)

# Per-WTG summary. Aggregate deviation compares mean actual with mean reference.
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
    ).reset_index()
)
wtg_summary["AggregateDeviation_pct"] = np.where(
    wtg_summary["MeanReferencePower_kW"] > 0,
    100 * (wtg_summary["MeanActualPower_kW"]
           - wtg_summary["MeanReferencePower_kW"])
    / wtg_summary["MeanReferencePower_kW"],
    np.nan
)
wtg_summary = wtg_summary.sort_values("AggregateDeviation_pct")

overall = pd.DataFrame([
    ["Total SCADA records", len(scada)],
    ["Records passing all required filters", int(mask.sum())],
    ["Records compared to reference curve", len(filtered)],
    ["WTGs compared", filtered["WTG"].nunique()],
    ["Reference curve min wind speed (m/s)", ref_min],
    ["Reference curve max wind speed (m/s)", ref_max],
    ["Reference curve air density", "1.070 kg/m³ (as stated in reference file)"],
    ["Filter 1", "PCurveStsAve = 100"],
    ["Filter 2", "WindSpeedStd < 1"],
    ["Filter 3", "-5 <= BladePitchStd <= 19"],
    ["Deviation sign", "Actual power - reference power; negative means below reference"],
], columns=["Metric", "Value"])

# Fleet power curve plot
plt.figure(figsize=(11, 7))
plt.plot(ref["ReferenceWindSpeed_mps"], ref["ReferencePower_kW"],
         linewidth=2.2, label="Reference Power Curve")
plt.scatter(filtered["WindSpeed_mps"], filtered["ActualPower_kW"],
            s=5, alpha=0.12, label="Filtered 10-min SCADA")
plt.plot(fleet_bins["MeanWindSpeed_mps"], fleet_bins["MeanActualPower_kW"],
         marker="o", linewidth=2, label="SCADA mean by 0.5 m/s bin")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Active Power (kW)")
plt.title("Jagaluru Site - Power Curve Analysis")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Jagaluru_Power_Curve_Comparison.png", dpi=180)
plt.close()

# Fleet deviation plot
plt.figure(figsize=(11, 6))
plt.axhline(0, linewidth=1)
plt.plot(fleet_bins["MeanWindSpeed_mps"], fleet_bins["MeanDeviation_kW"],
         marker="o", linewidth=2)
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Mean Deviation (Actual - Reference, kW)")
plt.title("Jagaluru Site - Power Deviation by Wind-Speed Bin")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Jagaluru_Power_Deviation.png", dpi=180)
plt.close()

details_cols = [
    "WTG", "DateTime", STATUS_COL, WIND_STD_COL, PITCH_STD_COL,
    "WindSpeed_mps", "ActualPower_kW", "ReferencePower_kW",
    "Deviation_kW", "Shortfall_kW", "Deviation_pct", "WindSpeedBin_mps"
]
with pd.ExcelWriter(OUTPUT_DIR / "Jagaluru_Power_Curve_Analysis.xlsx",
                    engine="openpyxl") as writer:
    overall.to_excel(writer, sheet_name="Overall Summary", index=False)
    filter_counts.to_excel(writer, sheet_name="Filter Counts", index=False)
    wtg_summary.to_excel(writer, sheet_name="WTG Summary", index=False)
    fleet_bins.to_excel(writer, sheet_name="Fleet Wind Bins", index=False)
    wtg_bins.to_excel(writer, sheet_name="WTG Wind Bins", index=False)
    filtered[details_cols].to_excel(
        writer, sheet_name="Filtered SCADA Details", index=False)
    ref.to_excel(writer, sheet_name="Reference Curve", index=False)

print("Power curve analysis completed.")
print(f"Rows passing filters: {int(mask.sum()):,}")
print(f"Rows compared with reference curve: {len(filtered):,}")
print(f"WTGs analysed: {filtered['WTG'].nunique()}")
print(f"Results saved in: {OUTPUT_DIR}")
