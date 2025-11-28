# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 13:06:29 2025

@author: Julia.Siebecker
"""

# =============================================================================
# Import three CSVs (wet, dry, dC)
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from PIL import Image


#%%
#file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/neu_hier_EntZippt/translation_fertig/00 kal 0513 SIM/retranslated/QuantReports/20250716_0513_SIM_Test_retranslated_2Substances/20250521_CalibrationData_AcqDateTime_local.csv"
#file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/neu_hier_EntZippt/translation_fertig/00 kal 0526 SIM/QuantReports/20250716_SIM_0526_test_2substances/20250521_CalibrationData_AcqDateTime_local.csv"
#file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/neu_hier_EntZippt/translation_fertig/00 kal 0611 SIM/QuantReports/20250716_Test_0611_SIM_2substances/20250521_CalibrationData_AcqDateTime_local.csv"

# df = pd.read_csv(file_path)  # all values are strings unless specified otherwise




file_direct_1 = "C:/Users/julia.siebecker/Documents\ATTO_Data/Daten Dry Season 2025/Calibration_Comparison/direct/QuantReports/DrySeason_ATTO_direct_1_V01/20250521_CalibrationData_AcqDateTime_local.csv"
file_direct_2 = "C:/Users/julia.siebecker/Documents\ATTO_Data/Daten Dry Season 2025/Calibration_Comparison/direct2/QuantReports/DrySeason_ATTO_direct_2_V01/20250521_CalibrationData_AcqDateTime_local.csv"
file_dry = "C:/Users/julia.siebecker/Documents\ATTO_Data/Daten Dry Season 2025/Calibration_Comparison/dry/QuantReports/DrySeason_ATTO_dry_V01/20250521_CalibrationData_AcqDateTime_local.csv"
file_wet = "C:/Users/julia.siebecker/Documents\ATTO_Data/Daten Dry Season 2025/Calibration_Comparison/wet/QuantReports/DrySeason_ATTO_wet_V01/20250521_CalibrationData_AcqDateTime_local.csv"

# =============================================================================
# Load CSVs
# =============================================================================
# Import all as strings unless specified otherwise
df_direct_1 = pd.read_csv(file_direct_1)
df_direct_2 = pd.read_csv(file_direct_2)
df_wet = pd.read_csv(file_wet)
df_dry = pd.read_csv(file_dry)

# =============================================================================
# Compound selection list
# =============================================================================
compoundlist = [
    "Chlorodifluoromethane",
    "Dichlorodifluoromethane",
    "Trichlorofluoromethane",
    "1,2-Dichloro-1,1,2,2-Tetrafluoroethane",
    "1,1,2-Trichloro-1,2,2-Trifluoroethane",
    "Chloromethane",
    "Chloroform",
    "Tetrachlormethane",
    "Isoprene"
]

# =============================================================================
# Cleaning and deduplication for each dataframe
# =============================================================================
def prepare_df(df):
    """Clean and rename columns like in the original single-file script."""
    df = df[df['Unnamed: 4'] != 'Name']
    df = df.dropna(subset=['Unnamed: 4'])
    deduped_df = df.drop_duplicates(subset=['Unnamed: 4'])

    # Extract relevant columns
    result_df = deduped_df[['Unnamed: 4', 'Textbox2', 'Textbox2.1', 'Textbox3']].copy()
    result_df.columns = ['Chemical Name', 'Formula', 'R²', 'Area']
    result_df.reset_index(drop=True, inplace=True)

    # === Add slope and intercept ===
    def extract_slope_intercept(equation):
        match = re.match(r'y\s*=\s*([\d\.\-]+)\s*\*\s*x\s*([\+\-])\s*([\d\.\-]+)', str(equation))
        if match:
            slope = float(match.group(1))
            sign = match.group(2)
            intercept = float(match.group(3))
            if sign == '-':
                intercept = -intercept
            return slope, intercept
        else:
            return None, None

    df['Slope'], df['Intercept'] = zip(*df['Textbox2'].apply(extract_slope_intercept))
    df['Slope'] = df['Slope'].apply(lambda x: f"{x:.6f}" if x is not None else None)
    df['Intercept'] = df['Intercept'].apply(lambda x: f"{x:.6f}" if x is not None else None)

    result_df = df[['Unnamed: 4', 'Textbox2', 'Textbox2.1', 'Textbox3', 'Slope', 'Intercept']].copy()
    result_df.columns = ['Chemical Name', 'Equation', 'R²', 'Area', 'Slope', 'Intercept']
    result_df.reset_index(drop=True, inplace=True)

    df = df.rename(columns={
        'Unnamed: 0': 'datafile_name',
        'Textbox5': 'time',
        'Unnamed: 2': 'type',
        'Unnamed: 3': 'something',
        'Unnamed: 4': 'analyte_col',
        'Unnamed: 5': 'rt',
        'Textbox2': 'geradengleichung',
        'Textbox2.1': 'r^2',
        'Textbox3': 'area',
        'Textbox4': 'level',
        'Textbox4.1': 'exp_conc'
    })

    return df, result_df


df_direct_1, result_direct_1 = prepare_df(df_direct_1)
df_direct_2, result_direct_2 = prepare_df(df_direct_2)
df_wet, result_wet = prepare_df(df_wet)
df_dry, result_dry = prepare_df(df_dry)

# =============================================================================
# Plot calibration curves: combined view per analyte
# =============================================================================
analyte_col = 'analyte_col'
x_col = 'exp_conc'
y_col = 'area'
datafile_col = 'datafile_name'

# Use date from first dataset for folder name
ordnerbasis_raw = str(df_direct_1.iloc[0, 0])
datumsteil = ordnerbasis_raw[:8]
ordnername = f"{datumsteil}_Kalibrierung_SIM_Comparison_TestScript"
os.makedirs(ordnername, exist_ok=True)

# Create dictionary for easier loop
dfs = {
    "Direct_1": df_direct_1,
    "Direct_2": df_direct_2,
    "Wet": df_wet,
    "Dry": df_dry
}

# Collect analyte names only from the defined list
all_analytes = sorted(
    set(df_direct_1[analyte_col].unique())
    | set(df_direct_2[analyte_col].unique())
    | set(df_wet[analyte_col].unique())
    | set(df_dry[analyte_col].unique())
)
selected_analytes = [a for a in all_analytes if str(a).strip().lower() in [c.lower() for c in compoundlist]]

# Loop over selected analytes and plot calibration curves
for analyte_name in selected_analytes:
    plt.figure(figsize=(7, 5))

    for label, df_current in dfs.items():
        group = df_current[df_current[analyte_col] == analyte_name]

        # Nur Kalibrationsdateien (direct, wet, dry)
        cal_group = group[group[datafile_col].str.contains(r'_(directCa(l)?_|wetCal_|dryCal_)', 
                                                          case=False, na=False, regex=True)]

        x = pd.to_numeric(cal_group[x_col], errors='coerce')
        y = pd.to_numeric(cal_group[y_col], errors='coerce')
        valid = x.notna() & y.notna()
        x = x[valid].values
        y = y[valid].values

        if len(x) < 2:
            continue  # Not enough data points for regression

        # Regression
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        eq = f"y = {slope:.2f}x + {intercept:.2f}"

        # Plot
        plt.scatter(x, y, label=f"{label} data")
        plt.plot(x, y_pred, label=f"{label}: {eq}, R²={r2:.4f}")

    plt.title(analyte_name)
    plt.xlabel("Concentration [ppb]")
    plt.ylabel("Area")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    dateiname = f"{datumsteil}_{analyte_name}_SIM_combined_linear.png"
    pfad = os.path.join(ordnername, dateiname)
    plt.savefig(pfad, bbox_inches='tight')
    plt.close()
