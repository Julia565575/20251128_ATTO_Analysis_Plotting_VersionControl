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

file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/Trial_Workflow/QuantReports/Translated_workflow_badIntegrations_forPythonScript/20250521_CalibrationData_AcqDateTime_notgiven_whatTime.csv"  # ← Update this to your actual path

df = pd.read_csv(file_path)  # all values are strings unless specified otherwise


#%%%
# =============================================================================
# relevant für erstellen deduped df
# =============================================================================

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
    plt.scatter(x, y)  # Nur Kalibrierpunkte!     , label="Kalibrierpunkte"
    plt.plot(x, y_pred, color='red', label=f"{gleichung}\n$R^2$ = {r2:.4f}")
    plt.title(analyt_name)
    plt.xlabel("Concentration [ppb]")
    plt.ylabel("Area")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Dateiname & Pfad
    dateiname = f"{datumsteil}_{analyt_name}_nurCalibrierpunkte3.png"
    pfad = os.path.join(ordnername, dateiname)

    # Plot speichern & schließen
    plt.savefig(pfad, bbox_inches='tight')
    plt.show()
    plt.close()

    # # Ergebnisse in df eintragen (nur für Kalibrierpunkte)
    # df.loc[cal_group.index[valid], 'Geradengleichung_Python'] = gleichung
    # df.loc[cal_group.index[valid], 'Slope_Python'] = slope
    # df.loc[cal_group.index[valid], 'Intercept_Python'] = intercept
    # df.loc[cal_group.index[valid], 'R^2_Python'] = r2


    # Ergebnisse in df eintragen – für alle Zeilen mit diesem Analyt
    df.loc[df[analyt_col] == analyt_name, 'Geradengleichung_Python'] = gleichung
    df.loc[df[analyt_col] == analyt_name, 'Slope_Python'] = slope
    df.loc[df[analyt_col] == analyt_name, 'Intercept_Python'] = intercept
    df.loc[df[analyt_col] == analyt_name, 'R^2_Python'] = r2

#%%
# =============================================================================
# Korrekturfunktion aus den dC Proben plotten und errechnen.
# =============================================================================
# Zeit als formatierten String (yyyy-mm-dd HH:MM) speichern
df['time_formated'] = pd.to_datetime(df['time'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d %H:%M')


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
    # 👉 Hier dein Wunschdatum eintragen (als Startpunkt für t = 0)
    referenz_zeitpunkt_str = "2025-04-30 10:36"
    t0 = pd.to_datetime(referenz_zeitpunkt_str)
    
    # Zeit relativ zum gewählten Zeitpunkt (in Sekunden)
    x_seconds = (x_time - t0).dt.total_seconds()
    
    # Lineare Regression: Fläche über Zeit (relativ)
    slope, intercept, r_value, p_value, std_err = linregress(x_seconds, y_area)
    
    # Fit-Linie berechnen
    x_fit = np.linspace(x_seconds.min(), x_seconds.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Zeitachse für Plot zurück in datetime (damit die Achse richtig beschriftet ist)
    x_fit_datetime = t0 + pd.to_timedelta(x_fit, unit='s')
    
    
    if len(x_time) < 2:
        continue

    sorted_indices = np.argsort(x_time)
    x_time = x_time.iloc[sorted_indices]
    y_area = y_area.iloc[sorted_indices]

    # Plot erstellen
    plt.figure(figsize=(6, 4))
    plt.scatter(x_time, y_area, color='blue') #, label="Datenpunkte"
    # plt.plot(x_fit_datetime, y_fit, color='red',
    #          label=f"Linearer Fit:\ny = {slope:.2f}·t + {intercept:.0f}\n(R² = {r_value**2:.3f})")
    plt.plot(x_fit_datetime, y_fit, color='red',
         label=f"y = {slope:.5f}·t + {intercept:.0f}\nR² = {r_value**2:.3f}")      #Linearer Fit:\n
    # Weitere Legendenbeschreibung: y_{t_0} = 2025-04-30 10:36
    plt.plot([], [], ' ', label=rf"$y_{{t_0}}$ = {referenz_zeitpunkt_str}")
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
    dateiname = f"{datumsteil}_{analyt_name}_Korrekturfunktion3.png"
    pfad = os.path.join(ordnername, dateiname)
    plt.savefig(pfad, bbox_inches='tight')
    plt.show()
    plt.close()

    # Fit-Gleichung als String
    formel_str = f"y = {slope:.5f}·t + {intercept:.0f}"
    r2_str = f"{r_value**2:.3f}"

    # Neue Spalten in den Original-DataFrame eintragen
    df.loc[df[analyt_col] == analyt_name, "Korrekturfunktion"] = formel_str
    df.loc[df[analyt_col] == analyt_name, "Korr_Fkt_R^2"] = r2_str



#%%
# =============================================================================
# 1. hier soll in dem df jede zeile iteriert werden, je nach analyt der Intercept_Python (plot 1) existiert in Intercept_Python
# 2. je nach Uhrzeit die entsprechende fläche des Analyten (plot 2) genutzt werden ist Korrekturfunktion
# 3. die Gerade durch die, die beiden Punkte berechnet werden
# 4. dann soll für die Fläche des Analyten die Konz bestimmt werden
# =============================================================================

#








#%%
# =============================================================================
# menge auftragen über zeitverlauf (proben)
# =============================================================================
