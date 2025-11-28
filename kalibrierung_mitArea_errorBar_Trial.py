# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:49:00 2025

@author: Julia.Siebecker
"""
import pandas as pd


# #pandas.read_csv(C:\Users\julia.siebecker\Documents\ATTO_Data\2025040_alles Cal bisher\QuantReports\20250430_kompletteCal_20250513_2_TXTandCSV_ResponseVSarea\20250513_ReportBuilder_suggested_corrArea)

# #i#mport pandas as pd

# # Define the full path to your CSV file
# file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/2025040_alles Cal bisher/QuantReports/20250520_forPt/20250519_CalibrationData.csv"  # ← Update this to your actual path

# # Force specific columns to be read as text (replace 'col1', 'col2' with your actual column names)
# df = pd.read_csv(file_path)

# # Optional: Convert those string values to float if needed later
# #df["col1"] = df["col1"].astype(float)

# # Print a preview
# print(df.head())


# %%
# =============================================================================
# 
# =============================================================================
import pandas as pd
import re
#file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/2025040_alles Cal bisher/QuantReports/20250520_forPt/20250520_CalibrationData_test_estConc.csv"  # ← Update this to your actual path
file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/2025040_alles Cal bisher/QuantReports/20250520_forPt/20250521_CalibrationData_AcqDateTime_in_local.csv"  # ← Update this to your actual path


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
result_df = deduped_df[['Unnamed: 4', 'Textbox2', 'Textbox2.1', 'Textbox3']].copy()

# Rename columns for clarity
result_df.columns = ['Chemical Name', 'Formula', 'R²', 'Area']

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
df['Slope'], df['Intercept'] = zip(*df['Textbox2'].apply(extract_slope_intercept))
df['Slope'] = df['Slope'].apply(lambda x: f"{x:.6f}" if x is not None else None)
df['Intercept'] = df['Intercept'].apply(lambda x: f"{x:.6f}" if x is not None else None)

# Final DataFrame with slope and intercept
result_df = df[['Unnamed: 4', 'Textbox2', 'Textbox2.1', 'Textbox3', 'Slope', 'Intercept']].copy()
result_df.columns = ['Chemical Name', 'Equation', 'R²', 'Area', 'Slope', 'Intercept']
result_df.reset_index(drop=True, inplace=True)

print(result_df)


#%%
# =============================================================================
# plotten der datenpunkte und kalgeraden
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Spaltenangaben ===
analyt_col = 'Unnamed: 4'       # Spalte mit Analytennamen
x_col = 'Textbox4.1'            # Konzentration [ppb]
y_col = 'Textbox3'              # Area

# === Zielordner vorbereiten ===
ordnerbasis_raw = str(df.iloc[0, 0])       # z.B. Datum aus Spalte 0
datumsteil = ordnerbasis_raw[:8]           # nur die ersten 8 Zeichen
ordnername = f"{datumsteil}_Kalibrierung"
os.makedirs(ordnername, exist_ok=True)

# Neue Spalten im df anlegen
df['Geradengleichung_Python'] = np.nan
df['Slope_Python'] = np.nan
df['Intercept_Python'] = np.nan
df['R^2_Python'] = np.nan

# Für jeden Analyt eine Regression durchführen
for analyt_name, group in df.groupby(analyt_col):
    # x und y aufbereiten
    x = pd.to_numeric(group[x_col], errors='coerce')
    y = pd.to_numeric(group[y_col], errors='coerce')
    valid = x.notna() & y.notna()
    x = x[valid].values
    y = y[valid].values

    if len(x) < 2:
        continue  # Nicht genug Daten für Regression

    # Regressionsparameter berechnen
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    gleichung = f"y = {slope:.2f}x + {intercept:.2f}"

    # Plot erstellen
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, label="Messwerte")
    plt.plot(x, y_pred, color='red', label=f"{gleichung}\n$R^2$ = {r2:.4f}")
    plt.title(analyt_name)
    plt.xlabel("Concentration [ppb]")
    plt.ylabel("Area")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Dateiname & Pfad
    dateiname = f"{datumsteil}_{analyt_name}.png"
    pfad = os.path.join(ordnername, dateiname)

    # Plot speichern & schließen
    plt.savefig(pfad)
    plt.close()

    # Ergebnisse in df eintragen
    df.loc[group.index[valid], 'Geradengleichung_Python'] = gleichung
    df.loc[group.index[valid], 'Slope_Python'] = slope
    df.loc[group.index[valid], 'Intercept_Python'] = intercept
    df.loc[group.index[valid], 'R^2_Python'] = r2










#%%

# =============================================================================
# plot der geraden mit der Steigung aus MassHUnter
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

x = np.linspace(0, 100, 100)

# Spalten 7, 8, 9 als float sicherstellen (R², Slope, Intercept)
for col in [7, 8, 9]:
    deduped_df.iloc[:, col] = deduped_df.iloc[:, col].apply(lambda x: float(str(x).strip()))

ordnerbasis_raw = str(deduped_df.iloc[0, 0])  # als string sichern
datumsteil = ordnerbasis_raw[:8]              # nur die ersten 8 Zeichen
ordnername = f"{datumsteil}_Kalibrierung"

os.makedirs(ordnername, exist_ok=True)

# Plots erzeugen und speichern
for index, row in deduped_df.iterrows():
    analyte_name = str(row[4]).strip()  # Analytenname (Spalte 4)
    legende = str(row[6]).strip()       # Legendentext (Spalte 6)
    r_squared = float(row[7])           # R²-Wert (Spalte 7)
    slope = float(row[8])               # Steigung (Spalte 8)
    intercept = float(row[9])           # Y-Achsenabschnitt (Spalte 9)
    basisname = str(row[1]).strip()     # Spalte 1 für Dateiname

    y = slope * x + intercept

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label=f"{legende}\n$R^2$ = {r_squared:.3f}")
    plt.title(analyte_name)
    plt.xlabel("Konz [ppb]")
    plt.ylabel("Area")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Dateipfad definieren
    dateiname = f"{datumsteil}_{analyte_name}.png"
    pfad = os.path.join(ordnername, dateiname)

    # Plot speichern
    plt.savefig(pfad)
    plt.close()



#%%
for analyt_name, group in df.groupby(analyt_col):
    # Extract and clean relevant columns
    group['Konz'] = pd.to_numeric(group['Textbox4'], errors='coerce')   # Calibration level
    group['Area'] = pd.to_numeric(group['Textbox3'], errors='coerce')   # Area
    valid = group['Konz'].notna() & group['Area'].notna()
    group = group[valid]

    if group.empty or len(group) < 2:
        continue

    # Group by calibration level and calculate stats
    stats = group.groupby('Konz')['Area'].agg(['mean', 'std', 'count']).reset_index()
    stats['se'] = stats['std'] / stats['count']**0.5  # Standard error
    stats.columns = ['Konz', 'Mean Area', 'Std Dev', 'N', 'SE']

    # Regression
    slope, intercept = np.polyfit(group['Konz'], group['Area'], 1)
    y_pred = slope * group['Konz'] + intercept
    r2 = 1 - np.sum((group['Area'] - y_pred) ** 2) / np.sum((group['Area'] - np.mean(group['Area'])) ** 2)
    gleichung = f"y = {slope:.2f}x + {intercept:.2f}"

    # Plot
    plt.figure(figsize=(6, 4))
    plt.errorbar(stats['Konz'], stats['Mean Area'], yerr=stats['SE'],
                 fmt='o', capsize=3, label='Messwerte ± SE')
    x_fit = np.linspace(stats['Konz'].min(), stats['Konz'].max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', label=f"{gleichung}\n$R^2$ = {r2:.4f}")

    plt.title(analyt_name)
    plt.xlabel("Concentration [ppb]")
    plt.ylabel("Area")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    dateiname = f"{datumsteil}_{analyt_name}_errorbars.png"
    pfad = os.path.join(ordnername, dateiname)
    plt.savefig(pfad)
    plt.close()

    # Optional: Add standard deviation to the original group
    for idx, row in stats.iterrows():
        df.loc[(df[analyt_col] == analyt_name) & (pd.to_numeric(df['Textbox4'], errors='coerce') == row['Konz']), 'Std_Dev'] = row['Std Dev']


#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Ensure correct dtypes
df['Textbox4'] = pd.to_numeric(df['Textbox4'], errors='coerce')  # Calibration level
df['Textbox3'] = pd.to_numeric(df['Textbox3'], errors='coerce')  # Area

# Make sure output folder exists
os.makedirs(ordnername, exist_ok=True)

# Initialize columns if not already present
df['Mean_Area'] = np.nan
df['Std_Dev'] = np.nan
df['SE'] = np.nan

# Loop over each analyte
for analyt_name, group in df.groupby(analyt_col):
    group = group.dropna(subset=['Textbox4', 'Textbox3'])
    if len(group) < 2:
        continue

    # Calculate stats
    stats = group.groupby('Textbox4')['Textbox3'].agg(['mean', 'std', 'count']).reset_index()
    stats['se'] = stats['std'] / stats['count']**0.5
    stats.columns = ['Konz', 'Mean_Area', 'Std_Dev', 'N', 'SE']

    if len(stats) < 2:
        continue  # Not enough points to draw a line

    # Linear regression
    slope, intercept = np.polyfit(stats['Konz'], stats['Mean_Area'], 1)
    y_pred = slope * stats['Konz'] + intercept
    r2 = 1 - np.sum((stats['Mean_Area'] - y_pred) ** 2) / np.sum((stats['Mean_Area'] - np.mean(stats['Mean_Area'])) ** 2)
    gleichung = f"y = {slope:.2f}x + {intercept:.2f}"

    # Plot with error bars
    plt.figure(figsize=(6, 4))
    plt.errorbar(stats['Konz'], stats['Mean_Area'], yerr=stats['SE'], fmt='o', capsize=3,
                 label='Mittelwert ± SE', color='blue')
    x_fit = np.linspace(stats['Konz'].min(), stats['Konz'].max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', label=f"{gleichung}\n$R^2$ = {r2:.4f}")

    plt.title(analyt_name)
    plt.xlabel("Concentration [ppb]")
    plt.ylabel("Area")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    dateiname = f"{datumsteil}_{analyt_name}_errorbars.png"
    pfad = os.path.join(ordnername, dateiname)
    plt.savefig(pfad)
    plt.close()

    # Save stats back into df for each matching entry
    for _, row in stats.iterrows():
        mask = (df[analyt_col] == analyt_name) & (df['Textbox4'] == row['Konz'])
        df.loc[mask, 'Mean_Area'] = row['Mean_Area']
        df.loc[mask, 'Std_Dev'] = row['Std_Dev']
        df.loc[mask, 'SE'] = row['SE']
