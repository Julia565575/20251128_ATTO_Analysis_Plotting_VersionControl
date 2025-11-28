# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:49:00 2025

@author: Julia.Siebecker
"""
import pandas as pd


#pandas.read_csv(C:\Users\julia.siebecker\Documents\ATTO_Data\2025040_alles Cal bisher\QuantReports\20250430_kompletteCal_20250513_2_TXTandCSV_ResponseVSarea\20250513_ReportBuilder_suggested_corrArea)

#i#mport pandas as pd

# Define the full path to your CSV file
file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/2025040_alles Cal bisher/QuantReports/20250430_kompletteCal_20250513_2_TXTandCSV_ResponseVSarea/20250513_ReportBuilder_suggested_corrArea.csv"  # ← Update this to your actual path

# Force specific columns to be read as text (replace 'col1', 'col2' with your actual column names)
df = pd.read_csv(file_path)

# Optional: Convert those string values to float if needed later
#df["col1"] = df["col1"].astype(float)

# Print a preview
print(df.head())


# %%
# =============================================================================
# 
# =============================================================================
import pandas as pd
import re
file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/2025040_alles Cal bisher/QuantReports/20250430_kompletteCal_20250514_testCalibrationExport/20250513_test_forCalData_forExcel4_from2headerSetTo1.csv"  # ← Update this to your actual path

# Step 1: Read the CSV
df = pd.read_csv(file_path)  # all values are strings unless specified otherwise







#%%%
# =============================================================================
# relevant für erstellen deduped df
# =============================================================================
#== funktionier
# Drop the header row if it's repeated within the dataset
df = df[df['Unnamed: 4'] != 'Name']

# Drop rows where chemical name is NaN (not part of the actual data)
df = df.dropna(subset=['Unnamed: 4'])

# Remove duplicates based on chemical name
deduped_df = df.drop_duplicates(subset=['Unnamed: 4'])

# Extract the relevant columns
result_df = deduped_df[['Unnamed: 4', 'Unnamed: 6', 'Unnamed: 7']].copy()

# Rename columns for clarity
result_df.columns = ['Chemical Name', 'Formula', 'R²']

# Reset index for a clean DataFrame
result_df.reset_index(drop=True, inplace=True)

print(result_df)


#%%
# =============================================================================
# im deduped df die slope und intercept hinzufügen. mit allen decimals
# =============================================================================
# Function to extract slope and intercept
def extract_slope_intercept(equation):
    match = re.match(r'y\s*=\s*([\d\.\-]+)\s*\*\s*x\s*([\+\-])\s*([\d\.\-]+)', equation)
    if match:
        slope = float(match.group(1))
        sign = match.group(2)
        intercept = float(match.group(3))
        if sign == '-':
            intercept = -intercept
        return slope, intercept
    else:
        return None, None

# Extract slope and intercept
deduped_df['Slope'], deduped_df['Intercept'] = zip(*deduped_df['Unnamed: 6'].apply(extract_slope_intercept))
deduped_df['Slope'] = deduped_df['Slope'].apply(lambda x: f"{x:.6f}" if x is not None else None)
deduped_df['Intercept'] = deduped_df['Intercept'].apply(lambda x: f"{x:.6f}" if x is not None else None)

# Final DataFrame with slope and intercept
result_df = deduped_df[['Unnamed: 4', 'Unnamed: 6', 'Unnamed: 7', 'Slope', 'Intercept']].copy()
result_df.columns = ['Chemical Name', 'Equation', 'R²', 'Slope', 'Intercept']
result_df.reset_index(drop=True, inplace=True)

print(result_df)
#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.linspace(0, 100, 100)

# Spalten, die als float gebraucht werden: 7,8,9 (R², Slope, Intercept)
# for col in [8, 9]:
#     deduped_df.iloc[:, col] = deduped_df.iloc[:, col].astype(float)
for col in [7, 8, 9]:
    deduped_df.iloc[:, col] = deduped_df.iloc[:, col].apply(lambda x: float(str(x).strip()))


for index, row in deduped_df.iterrows():
    analyte_name = str(row[4])    # Analytenname (Spalte 4)
    legende = str(row[6])         # Legendentext (Spalte 6)
    r_squared = float(row[7])     # R²-Wert (Spalte 7)
    slope = float(row[8])         # Steigung (Spalte 8)
    intercept = float(row[9])     # Y-Achsenabschnitt (Spalte 9)

    y = slope * x + intercept

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label=f"{legende}\n$R^2$ = {r_squared:.3f}")
    plt.title(analyte_name)
    plt.xlabel("Konz [ppb]")
    plt.ylabel("Area")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



