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
#file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/2025040_alles Cal bisher/QuantReports/20250520_forPt/20250521_CalibrationData_AcqDateTime_in_local.csv"  # ← Update this to your actual path
# file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/Trial_Workflow/QuantReports/Translated_workflow_badIntegrations_forPythonScript/20250521_CalibrationData_AcqDateTime_in_local.csv"  # ← Update this to your actual path

# # Step 1: Read the CSV
# df = pd.read_csv(file_path)  # all values are strings unless specified otherwise


file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/Trial_Workflow/QuantReports/Translated_workflow_badIntegrations_forPythonScript/20250521_CalibrationData_AcqDateTime_notgiven_whatTime.csv"  # ← Update this to your actual path

df = pd.read_csv(file_path)  # all values are strings unless specified otherwise

# file_path3 = "C:/Users/julia.siebecker/Documents/ATTO_Data/Trial_Workflow/QuantReports/Translated_workflow_badIntegrations_forPythonScript/20250521_CalibrationData_AcqDateTime_local.csv"  # ← Update this to your actual path

# df3 = pd.read_csv(file_path3)  # all values are strings unless specified otherwise


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


# #%%
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# # === Spaltenangaben ===
# analyt_col = 'Unnamed: 4'       # Spalte mit Analytennamen
# x_col = 'Textbox4.1'            # Konzentration [ppb]
# y_col = 'Textbox3'              # Area
# data_name= 'Unnamed:0'

# # === Zielordner vorbereiten ===
# ordnerbasis_raw = str(df.iloc[0, 0])       # z.B. Datum aus Spalte 0
# datumsteil = ordnerbasis_raw[:8]           # nur die ersten 8 Zeichen
# ordnername = f"{datumsteil}_Kalibrierung"
# os.makedirs(ordnername, exist_ok=True)

# # Neue Spalten im df anlegen
# df['Geradengleichung_Python'] = np.nan
# df['Slope_Python'] = np.nan
# df['Intercept_Python'] = np.nan
# df['R^2_Python'] = np.nan

# # Für jeden Analyt eine Regression durchführen
# for analyt_name, group in df.groupby(analyt_col):
#     # Überprüfen, ob der Name den Teil _directCal_ enthält
    
#     if '_directCal_' in data_name:    
            
#         # x und y aufbereiten
#         x = pd.to_numeric(group[x_col], errors='coerce')
#         y = pd.to_numeric(group[y_col], errors='coerce')
#         valid = x.notna() & y.notna()
#         x = x[valid].values
#         y = y[valid].values


#         # Regressionsparameter berechnen
#         slope, intercept = np.polyfit(x, y, 1)
#         y_pred = slope * x + intercept
#         r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
#         gleichung = f"y = {slope:.2f}x + {intercept:.2f}"


#         # Plot erstellen
#         plt.figure(figsize=(6, 4))
#         plt.scatter(x, y, label="Messwerte")
#         plt.plot(x, y_pred, color='red', label=f"{gleichung}\n$R^2$ = {r2:.4f}")
#         plt.title(analyt_name)
#         plt.xlabel("Concentration [ppb]")
#         plt.ylabel("Area")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
    
#         # Dateiname & Pfad
#         dateiname = f"{datumsteil}_{analyt_name}_alleCalMessungen_Test_filteresDataset.png"
#         pfad = os.path.join(ordnername, dateiname)
    
#         # Plot speichern & schließen
#         plt.savefig(pfad)
#         plt.show()
#         plt.close()
    
#         # Ergebnisse in df eintragen
#         df.loc[group.index[valid], 'Geradengleichung_Python'] = gleichung
#         df.loc[group.index[valid], 'Slope_Python'] = slope
#         df.loc[group.index[valid], 'Intercept_Python'] = intercept
#         df.loc[group.index[valid], 'R^2_Python'] = r2


#%%
# =============================================================================
# plotten der datenpunkte und kalgeraden
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Umbenennen der Spalte 'Unnamed: 4' in 'Analyt_col'
df = df.rename(columns={'Unnamed: 0': 'datafile_name'})
df = df.rename(columns={'Textbox1': 'time'})
df = df.rename(columns={'Unnamed: 2': 'type'})
df = df.rename(columns={'Unnamed: 3': 'something'})
df = df.rename(columns={'Unnamed: 4': 'analyte_col'})
df = df.rename(columns={'Unnamed: 5': 'rt'})
df = df.rename(columns={'Textbox2': 'geradengleichung'})
df = df.rename(columns={'Textbox2.1': 'r^2'})
df = df.rename(columns={'Textbox3': 'area'})
df = df.rename(columns={'Textbox4': 'level'})
df = df.rename(columns={'Textbox4.1': 'exp_conc'})



# # Filtere die Kalibrierproben, die ..._directCal_... im Namen enthalten
# kalibrierproben = df[df['datafile_name'].str.contains('_directCal_')]

# # Filtere die Proben, die ..._dC_... im Namen enthalten
# dC_Korrektur = df[df['datafile_name'].str.contains('_dC_')]

# # Entferne die nicht aufzunehmenden Proben aus den Kalibrierproben
# kalibrierproben = kalibrierproben[~kalibrierproben['datafile_name'].isin(dC_Korrektur['datafile_name'])]
# === Spaltenangaben ===
analyt_col = 'analyte_col'       # Spalte mit Analytennamen
x_col = 'exp_conc'               # Konzentration [ppb]
y_col = 'area'                   # Area
datafile_col = 'datafile_name'   # Spalte mit Dateinamen

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
    # Nur Kalibrierpunkte auswählen
    cal_group = group[group[datafile_col].str.contains('_directCal_', na=False)]

    # x und y aufbereiten
    x = pd.to_numeric(cal_group[x_col], errors='coerce')
    y = pd.to_numeric(cal_group[y_col], errors='coerce')
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
    plt.scatter(x, y, label="Kalibrierpunkte")  # Nur Kalibrierpunkte!
    plt.plot(x, y_pred, color='red', label=f"{gleichung}\n$R^2$ = {r2:.4f}")
    plt.title(analyt_name)
    plt.xlabel("Concentration [ppb]")
    plt.ylabel("Area")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Dateiname & Pfad
    dateiname = f"{datumsteil}_{analyt_name}_nurCalibrierpunkte.png"
    pfad = os.path.join(ordnername, dateiname)

    # Plot speichern & schließen
    plt.savefig(pfad, bbox_inches='tight')
    plt.show()
    plt.close()

    # Ergebnisse in df eintragen (nur für Kalibrierpunkte)
    df.loc[cal_group.index[valid], 'Geradengleichung_Python'] = gleichung
    df.loc[cal_group.index[valid], 'Slope_Python'] = slope
    df.loc[cal_group.index[valid], 'Intercept_Python'] = intercept
    df.loc[cal_group.index[valid], 'R^2_Python'] = r2




#%%
# =============================================================================
# Korrekturfunktion aus den dC Proben plotten und errechnen.
# =============================================================================
# Für jeden Analyt eine Zeitreihe plotten
for analyt_name, group in df.groupby(analyt_col):
    dC_Korrektur = group[group[datafile_col].str.contains('_dC_', na=False)]
    dC_Korrektur['time'] = pd.to_datetime(dC_Korrektur['time'], dayfirst=True, errors='coerce') #, utc=True)
    
    # Zeit und Peakfläche NUR aus dC_Korrektur ziehen
    x_time = pd.to_datetime(dC_Korrektur['time'], errors='coerce') #, utc=True)
    y_area = pd.to_numeric(dC_Korrektur[y_col], errors='coerce')

    valid = x_time.notna() & y_area.notna()
    x_time = x_time[valid]
    y_area = y_area[valid]

    from scipy.stats import linregress

    # Lineare Regression: Zeit → Timestamp
    x_num = x_time.astype(np.int64) / 1e9  # Sekunden seit 1970
    slope, intercept, r_value, p_value, std_err = linregress(x_num, y_area)
    
    # Werte für Fit-Linie berechnen (nur innerhalb des Bereichs)
    x_fit = np.linspace(x_num.min(), x_num.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Zeitachsen zurück in datetime
    x_fit_datetime = pd.to_datetime(x_fit, unit='s') #, utc=True)
    
    if len(x_time) < 2:
        continue

    sorted_indices = np.argsort(x_time)
    x_time = x_time.iloc[sorted_indices]
    y_area = y_area.iloc[sorted_indices]

    # Plot erstellen
    plt.figure(figsize=(6, 4))
    plt.scatter(x_time, y_area, color='blue', label="Datenpunkte")
    # plt.plot(x_fit_datetime, y_fit, color='red',
    #          label=f"Linearer Fit:\ny = {slope:.2f}·t + {intercept:.0f}\n(R² = {r_value**2:.3f})")
    plt.plot(x_fit_datetime, y_fit, color='red',
         label=f"Linearer Fit:\ny = {slope:.5f}·t + {intercept:.0f}\n(R² = {r_value**2:.3f})")      
    plt.title(f"{analyt_name} – Korrekturfunktion")
    plt.xlabel("Messzeitpunkt (notUTC)")
    plt.ylabel("Area")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    

    import matplotlib.dates as mdates

    # Find the range of the x axis (datetime)
    start_date = x_time.min().normalize()  # midnight of the first date
    end_date = x_time.max().normalize() + pd.Timedelta(days=1)  # one day after last date (to include full last day)
    
    # Create list of ticks at midnight and midday for every day in range
    tick_times = []
    current = start_date
    while current <= end_date:
        tick_times.append(current)                    # midnight
        tick_times.append(current + pd.Timedelta(hours=12))  # midday
        current += pd.Timedelta(days=1)
    
    # Filter ticks to only those within the actual range of your data (optional, but cleaner)
    tick_times = [t for t in tick_times if x_time.min() <= t <= x_time.max()]
    
    # Convert to matplotlib dates for ticks
    tick_times_num = mdates.date2num(tick_times)    
    
    ax = plt.gca()
    ax.set_xticks(tick_times)  # deine erzeugten Tick-Zeitpunkte
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # <– hier wird das Datumsformat inkl. Uhrzeit gesetzt
    plt.xticks(rotation=45)
    # Tick-Labels holen und rechtsbündig ausrichten
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    
    # Speichern
    dateiname = f"{datumsteil}_{analyt_name}_Korrekturfunktion.png"
    pfad = os.path.join(ordnername, dateiname)
    plt.savefig(pfad, bbox_inches='tight')
    plt.show()
    plt.close()





#%%
#cal_group = group[group[datafile_col].str.contains('_directCal_', na=False)]

# --- Korrekturfunktion: Verlauf der Peakfläche über die Zeit für jeden Analyten ---

# Sicherstellen, dass die Zeitspalte als datetime interpretiert wird


# # Für jeden Analyt eine Zeitreihe plotten
# for analyt_name, group in df.groupby(analyt_col):
#     dC_Korrektur = group[group[datafile_col].str.contains('_dC_', na=False)]
#     dC_Korrektur['time'] = pd.to_datetime(dC_Korrektur['time'], errors='coerce', utc=True)
#     # Zeit und Peakfläche
#     x_time = pd.to_datetime(group['time'], errors='coerce', utc=True)
#     y_area = pd.to_numeric(group[y_col], errors='coerce')

#     # Nur gültige Werte behalten
#     valid = x_time.notna() & y_area.notna()
#     x_time = x_time[valid]
#     y_area = y_area[valid]

#     if len(x_time) < 2:
#         continue  # Nicht genug Daten

#     # Sortieren nach Zeit (optional, für schöne Plots)
#     sorted_indices = np.argsort(x_time)
#     x_time = x_time.iloc[sorted_indices]
#     y_area = y_area.iloc[sorted_indices]

#     # Plot erstellen
#     plt.figure(figsize=(6, 4))
#     plt.plot(x_time, y_area, marker='o', linestyle='-', color='blue', label="Peakfläche über Zeit")
#     plt.title(f"{analyt_name} – Korrekturfunktion")
#     plt.xlabel("Messzeitpunkt (notUTC)")
#     plt.ylabel("Area")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
    
#     # X-Achse formatieren: nur tägliche Ticks um 12 Uhr
#     import matplotlib.dates as mdates
    
#     # Tägliche Ticks um 12:00 Uhr mittags
#     ax = plt.gca()
#     start_date = x_time.min().normalize()
#     end_date = x_time.max().normalize()
#     tick_dates = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC') + pd.Timedelta(hours=12)
    
#     ax.set_xticks(tick_dates)
#     ax.set_xlim(tick_dates.min() - pd.Timedelta(hours=12), tick_dates.max() + pd.Timedelta(hours=12))
#     ax.xaxis.set_major_locator(mdates.FixedLocator(tick_dates))  # Nur diese Ticks verwenden!
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y %H:%M'))
#     # Rotiere die Labels direkt am Axes-Objekt
#     for label in ax.get_xticklabels():
#         label.set_rotation(45)
#         label.set_ha('right')  # optional für bessere Positionierung
#     #plt.xticks(rotation=45)
        

#     # Dateiname & Pfad
#     dateiname = f"{datumsteil}_{analyt_name}_Korrekturfunktion_test1.png"
#     pfad = os.path.join(ordnername, dateiname)

#     # Plot speichern & schließen
#     plt.savefig(pfad)
#     plt.show()
#     plt.close()





