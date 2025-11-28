# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:16:52 2025

@author: julia.siebecker
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




file_wet = "C:/Users/julia.siebecker/Documents/ATTO_Data/neu_hier_EntZippt/translation_fertig/00 kal 0513 SIM/retranslated/QuantReports/20250716_0513_SIM_Test_retranslated_2Substances/20250521_CalibrationData_AcqDateTime_local.csv"
file_dry = "C:/Users/julia.siebecker/Documents/ATTO_Data/neu_hier_EntZippt/translation_fertig/00 kal 0526 SIM/QuantReports/20250716_SIM_0526_test_2substances/20250521_CalibrationData_AcqDateTime_local.csv"
file_dC = "C:/Users/julia.siebecker/Documents/ATTO_Data/neu_hier_EntZippt/translation_fertig/00 kal 0611 SIM/QuantReports/20250716_Test_0611_SIM_2substances/20250521_CalibrationData_AcqDateTime_local.csv"


# --- Add your file paths here ---
# file_wet = r"path_to_wet.csv"
# file_dry = r"path_to_dry.csv"
# file_dC  = r"path_to_dC.csv"

# Import all as strings unless specified otherwise
df_wet = pd.read_csv(file_wet)
df_dry = pd.read_csv(file_dry)
df_dC  = pd.read_csv(file_dC)


#%%%
# =============================================================================
# Cleaning and deduplication for each dataframe
# =============================================================================
def prepare_df(df):
    """Clean and rename columns like in the original single-file script."""
    # Drop repeated header lines
    df = df[df['Unnamed: 4'] != 'Name']
    df = df.dropna(subset=['Unnamed: 4'])

    # Drop duplicates
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

    # Rename columns for later processing
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


df_wet, result_wet = prepare_df(df_wet)
df_dry, result_dry = prepare_df(df_dry)
df_dC,  result_dC  = prepare_df(df_dC)


#%%%
# =============================================================================
# Plot calibration curves: combined view per analyte (wet, dry, dC)
# =============================================================================
analyte_col = 'analyte_col'
x_col = 'exp_conc'
y_col = 'area'
datafile_col = 'datafile_name'

# Use date from first dataset for folder name
ordnerbasis_raw = str(df_wet.iloc[0, 0])
datumsteil = ordnerbasis_raw[:8]
ordnername = f"{datumsteil}_Kalibrierung_SIM_Comparison_TestScript"    #################################################################################
os.makedirs(ordnername, exist_ok=True)

# Create dictionary for easier loop
dfs = {
    "WET": df_wet,
    "DRY": df_dry,
    "dC": df_dC
}

# Collect all analyte names
all_analytes = sorted(
    set(df_wet[analyte_col].unique())
    | set(df_dry[analyte_col].unique())
    | set(df_dC[analyte_col].unique())
)

 
# Loop over all analytes and plot all three calibration curves together
for analyte_name in all_analytes:
    plt.figure(figsize=(6, 4))

    for label, df_current in dfs.items():
        group = df_current[df_current[analyte_col] == analyte_name]
        cal_group = group[group[datafile_col].str.contains('_directCal_', na=False)]

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
    plt.show()
    plt.close()

    # Add metadata
    script_name = os.path.basename(__file__)
    with Image.open(pfad) as img:
        img.info['ScriptName'] = script_name
        img.save(pfad, "PNG")

print(f"✅ Combined calibration plots saved in '{ordnername}'")



#test change for git