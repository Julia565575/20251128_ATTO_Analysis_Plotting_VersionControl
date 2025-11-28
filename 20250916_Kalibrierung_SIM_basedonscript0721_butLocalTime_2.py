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

#file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/neu_hier_EntZippt/translation_fertig/00 kal 0513 SCAN/QuantReports/20250711_Kal0513_Test1_2Substances/20250711_CalibrationData_AcqDateTime_local.csv"

#file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/neu_hier_EntZippt/translation_fertig/00 kal 0430 SIM/QuantReports/20250718_SIM_0430_test_2substances_andFirstComparison_corrected/20250521_CalibrationData_AcqDateTime_local.csv"
file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/neu_hier_EntZippt/translation_fertig/00 kal 0513 SIM/retranslated/QuantReports/20250716_0513_SIM_Test_retranslated_2Substances/20250521_CalibrationData_AcqDateTime_local.csv"
#file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/neu_hier_EntZippt/translation_fertig/00 kal 0526 SIM/QuantReports/20250716_SIM_0526_test_2substances/20250521_CalibrationData_AcqDateTime_local.csv"
#file_path = "C:/Users/julia.siebecker/Documents/ATTO_Data/neu_hier_EntZippt/translation_fertig/00 kal 0611 SIM/QuantReports/20250716_Test_0611_SIM_2substances/20250521_CalibrationData_AcqDateTime_local.csv"


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
df = df.rename(columns={'Textbox5': 'time'})
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
analyte_col = 'analyte_col'       # Spalte mit Analytennamen
x_col = 'exp_conc'               # Konzentration [ppb]
y_col = 'area'                   # Area
datafile_col = 'datafile_name'   # Spalte mit Dateinamen

# === Zielordner vorbereiten ===
ordnerbasis_raw = str(df.iloc[0, 0])       # z.B. Datum aus Spalte 0
datumsteil = ordnerbasis_raw[:8]           # nur die ersten 8 Zeichen
ordnername = f"{datumsteil}_Kalibrierung_SIM"
os.makedirs(ordnername, exist_ok=True)

# Neue Spalten im df anlegen
df['Geradengleichung_Python'] = np.nan
df['Slope_Python'] = np.nan
df['Intercept_Python'] = np.nan
df['R^2_Python'] = np.nan

# Für jeden Analyt eine Regression durchführen
for analyt_name, group in df.groupby(analyte_col):
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
    dateiname = f"{datumsteil}_{analyt_name}_SIM_nurCalibrierpunkte_linear.png"
    pfad = os.path.join(ordnername, dateiname)

    # Plot speichern & schließen
    plt.savefig(pfad, bbox_inches='tight')
    plt.show()
    plt.close()
    # Add metadata to the image
    from PIL import Image
    # Automatically get the script name
    script_name = os.path.basename(__file__)
    with Image.open(pfad) as img:
        img.info['ScriptName'] = script_name
        img.save(pfad, "PNG")

    # # Ergebnisse in df eintragen (nur für Kalibrierpunkte)
    # df.loc[cal_group.index[valid], 'Geradengleichung_Python'] = gleichung
    # df.loc[cal_group.index[valid], 'Slope_Python'] = slope
    # df.loc[cal_group.index[valid], 'Intercept_Python'] = intercept
    # df.loc[cal_group.index[valid], 'R^2_Python'] = r2


    # Ergebnisse in df eintragen – für alle Zeilen mit diesem Analyt
    df.loc[df[analyte_col] == analyt_name, 'Geradengleichung_Python'] = gleichung
    df.loc[df[analyte_col] == analyt_name, 'Slope_Python'] = slope
    df.loc[df[analyte_col] == analyt_name, 'Intercept_Python'] = intercept
    df.loc[df[analyte_col] == analyt_name, 'R^2_Python'] = r2

#%%
# =============================================================================
# Korrekturfunktion aus den dC Proben plotten und errechnen.
# =============================================================================
# Zeit als formatierten String (yyyy-mm-dd HH:MM) speichern
# df['time_formated'] = pd.to_datetime(df['time'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
df['time_formated'] = pd.to_datetime(df['time'], dayfirst=True, errors='coerce')
# Create local time column (UTC - 4 hours)
df['time_formated_local'] = df['time_formated'] - pd.Timedelta(hours=4)


# Für jeden Analyt eine Zeitreihe plotten
for analyt_name, group in df.groupby(analyte_col):
    dC_Korrektur = group[group[datafile_col].str.contains('_dC_', na=False)]

    # Zeit und Peakfläche aus der bereits formatierten Spalte ziehen
    x_time = pd.to_datetime(dC_Korrektur['time_formated_local'], errors='coerce')
    y_area = pd.to_numeric(dC_Korrektur[y_col], errors='coerce')

    valid = x_time.notna() & y_area.notna()
    x_time = x_time[valid]
    y_area = y_area[valid]

    from scipy.stats import linregress
    referenz_zeitpunkt_str = "2025-05-13 20:14"         #UTC, because correction over time based on UTC Times. Time calc made for plots, after the calculation
    t0 = pd.to_datetime(referenz_zeitpunkt_str)

    # Zeit relativ zum gewählten Zeitpunkt (in Sekunden)
    x_seconds = (x_time - t0).dt.total_seconds()

    # Lineare Regression: Fläche über Zeit
    slope, intercept, r_value, p_value, std_err = linregress(x_seconds, y_area)

    # Fit-Linie berechnen
    x_fit = np.linspace(x_seconds.min(), x_seconds.max(), 100)
    y_fit = slope * x_fit + intercept
    x_fit_datetime = t0 + pd.to_timedelta(x_fit, unit='s')

    if len(x_time) < 2:
        continue

    sorted_indices = np.argsort(x_time)
    x_time = x_time.iloc[sorted_indices]
    y_area = y_area.iloc[sorted_indices]

    # Plot erstellen
    plt.figure(figsize=(6, 4))
    plt.scatter(x_time, y_area, color='blue')
    plt.plot(x_fit_datetime, y_fit, color='red',
             label=f"y = {slope:.5f}·t + {intercept:.0f}\nR² = {r_value**2:.3f}")
    plt.plot([], [], ' ', label=rf"$y_{{t_0}}$ = {referenz_zeitpunkt_str}")
    plt.title(f"{analyt_name} – Korrekturfunktion")
    plt.xlabel("Messzeitpunkt (notUTC)")
    plt.ylabel("Area")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    import matplotlib.dates as mdates
    start_date = x_time.min().normalize()
    end_date = x_time.max().normalize() + pd.Timedelta(days=1)

    tick_times = []
    current = start_date
    while current <= end_date:
        tick_times.append(current)
        tick_times.append(current + pd.Timedelta(hours=12))
        current += pd.Timedelta(days=1)

    tick_times = [t for t in tick_times if x_time.min() <= t <= x_time.max()]
    tick_times_num = mdates.date2num(tick_times)

    ax = plt.gca()
    ax.set_xticks(tick_times)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    # Speichern
    dateiname = f"{datumsteil}_{analyt_name}_SIM_Korrekturfunktion_linear.png"
    pfad = os.path.join(ordnername, dateiname)
    plt.savefig(pfad, bbox_inches='tight')
    plt.show()
    plt.close()

    # Fit-Gleichung als String
    formel_str = f"y = {slope:.5f}·t + {intercept:.0f}"
    r2_str = f"{r_value**2:.3f}"

    # Eintragen in DataFrame für alle Zeilen mit diesem Analyten
    df.loc[df[analyte_col] == analyt_name, "Korrekturfunktion"] = formel_str
    df.loc[df[analyte_col] == analyt_name, "Korr_Fkt_R^2"] = r2_str



#%%
# =============================================================================
# 1. hier soll in dem df jede zeile iteriert werden, je nach analyt der Intercept_Python (plot 1) existiert in Intercept_Python
# 2. je nach Uhrzeit die entsprechende fläche des Analyten (plot 2) genutzt werden ist Korrekturfunktion
# 3. die Gerade durch die, die beiden Punkte berechnet werden
# 4. dann soll für die Fläche des Analyten die Konz bestimmt werden
# =============================================================================


# =============================================================================
# 2. und 3. korr_Kalpoint berechnen und korr Kalfunktion berechnen
# =============================================================================
# Neue Spalte vorbereiten
df['korr_Kalpoint'] = np.nan

# # Sicherstellen, dass time_formated als datetime interpretiert wird
# df['time_formated'] = pd.to_datetime(df['time_formated'], errors='coerce')

# Funktion zur Berechnung des Korrekturwerts pro Zeile
def berechne_korr_zeitwert(row):
    fkt = row.get('Korrekturfunktion')
    zeitpunkt = row.get('time_formated')
    
    if pd.isna(fkt) or pd.isna(zeitpunkt):
        return np.nan
    
    try:
        # Extrahiere Steigung und Achsenabschnitt
        slope_str, intercept_str = fkt.split("·t +")
        slope = float(slope_str.split("= ")[1])
        intercept = float(intercept_str)
        
        # Zeitdifferenz in Sekunden
        delta_t = (zeitpunkt - t0).total_seconds()
        return slope * delta_t + intercept
    except:
        return np.nan

# Funktion anwenden
df['korr_Kalpoint'] = df.apply(berechne_korr_zeitwert, axis=1)

#EXP konz eintragen
# Neue Spalte initialisieren
df['expected_conc'] = np.nan

# Für jeden Analyt den exp_conc-Wert aus _dC_-Zeilen holen
for analyt_name, group in df.groupby('analyte_col'):
    # Nur _dC_ Zeilen dieses Analyten
    dC_group = group[group['datafile_name'].str.contains('_dC_', na=False)]
    
    # Falls keine _dC_-Messung vorhanden → überspringen
    if dC_group.empty:
        continue
    
    # Hole den erwarteten Konzentrationswert (sollte bei allen gleich sein)
    unique_exp = dC_group['exp_conc'].dropna().unique()
    
    if len(unique_exp) == 1:
        # Trage diesen Wert für alle Zeilen des Analyten ein
        df.loc[df['analyte_col'] == analyt_name, 'expected_conc'] = unique_exp[0]
    else:
        print(f"Achtung: Mehrere exp_conc-Werte gefunden für {analyt_name}: {unique_exp}")




# Neue Spalte für die korrigierte Kalibrierfunktion anlegen
df["korr_CalFunction"] = np.nan

# Über alle Zeilen iterieren
for idx, row in df.iterrows():
    try:
        intercept = float(row['Intercept_Python'])
        x1 = float(row['expected_conc'])
        y1 = float(row['korr_Kalpoint'])

        # Steigung berechnen: m = (y1 - y0) / (x1 - x0), wobei y0 = intercept bei x0 = 0
        m = (y1 - intercept) / x1
        b = intercept

        # Funktion als String speichern
        function_str = f"y = {m:.4f}x + {b:.2f}"
        df.at[idx, 'korr_CalFunction'] = function_str

    except (ValueError, TypeError, ZeroDivisionError):
        # Falls ein Wert fehlt oder ungültig ist: Feld leer lassen
        continue

# =============================================================================
# Calculate Conc
# =============================================================================

import re
df["Conc"] = np.nan  # Create new column for concentration

for idx, row in df.iterrows():
    try:
        area = float(row['area'])
        func_str = row['korr_CalFunction']
        
        if pd.notna(area) and isinstance(func_str, str):
            # Extract m and b using regex
            match = re.match(r"y\s*=\s*([-+]?[0-9]*\.?[0-9]+)x\s*\+\s*([-+]?[0-9]*\.?[0-9]+)", func_str)
            if match:
                m = float(match.group(1))
                b = float(match.group(2))

                # Calculate concentration
                conc = (area - b) / m
                df.at[idx, 'Conc'] = conc

    except Exception as e:
        continue  # Skip row if anything goes wrong




#%%
# =============================================================================
# conc auftragen über zeitverlauf (proben)
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd  # falls nicht schon importiert

# Stelle sicher, dass time_formated eine datetime-Spalte ist
df['time_formated'] = pd.to_datetime(df['time_formated'], errors='coerce')

# Create local time column (UTC - 4 hours)
df['time_formated_local'] = df['time_formated'] - pd.Timedelta(hours=4)


# === Neue Zuordnungen für Farben, Labels & Marker ===
farben = {
    '_0m_': 'green',
    '_80m_': 'red',
    '_320m_': 'blue'
}

label_dict = {
    '_0m_': '2 meters',
    '_80m_': '80 meters',
    '_320m_': '320 meters'
}

marker_dict = {
    '_0m_': 'o',
    '_80m_': 's',
    '_320m_': '^'
}

# Für jeden Analyt einzeln plotten
for analyt_name, group in df.groupby(analyte_col):
    plt.figure(figsize=(10, 6))
    
    for typ, farbe in farben.items():
        teilgruppe = group[group[datafile_col].str.contains(typ, na=False)]
        
        plt.scatter(
            teilgruppe['time_formated_local'],
            teilgruppe['Conc'],
            color=farbe,
            marker=marker_dict.get(typ, 'o'),  # Optional: verschiedene Marker
            label=label_dict.get(typ, typ),    # Freundliche Bezeichnung in der Legende
            alpha=0.7
        )

    # Titel, Achsenbeschriftungen, Legende
    plt.title(f"{analyt_name}", fontsize=18)
    plt.xlabel("date (local time)", fontsize=14)
    plt.ylabel("Calculated Concentration [ppb]", fontsize=14)
    plt.legend(title="Sample Type", fontsize=14, title_fontsize=16)  # Hier wird die Legende angezeigt
    plt.grid(True)
    plt.tight_layout()

    # =========================
    # X-Achsen-Formatierung
    # =========================
    x_time = pd.to_datetime(group['time_formated_local'], errors='coerce')
    x_time = x_time.dropna()

    if not x_time.empty:
        ax = plt.gca()

        # Set major ticks at midnight
        ax.xaxis.set_major_locator(mdates.DayLocator())

        # Show only the date (YYYY-MM-DD)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Rotate and align labels
        plt.xticks(rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')


    # Speichern
    dateiname = f"{datumsteil}_{analyt_name}_SIM_Proben_linear.png"
    pfad = os.path.join(ordnername, dateiname)
    plt.savefig(pfad, bbox_inches='tight')
    plt.show()
    plt.close()

#%%
# =============================================================================
# diurnal plot
# =============================================================================



import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd

# Ensure 'time_formated' is a datetime column
df['time_formated'] = pd.to_datetime(df['time_formated'], errors='coerce')

# New assignments for colors, labels & markers
farben = {
    '_0m_': 'green',
    '_80m_': 'red',
    '_320m_': 'blue'
}

label_dict = {
    '_0m_': '2 meters',
    '_80m_': '80 meters',
    '_320m_': '320 meters'
}

marker_dict = {
    '_0m_': 'o',
    '_80m_': 's',
    '_320m_': '^'
}

# Plot each analyte individually
for analyt_name, group in df.groupby(analyte_col):
    plt.figure(figsize=(10, 6))
    
    for typ, farbe in farben.items():
        teilgruppe = group[group[datafile_col].str.contains(typ, na=False)]
        
        # Convert time to hours since midnight
        hours_since_midnight = teilgruppe['time_formated_local'].dt.hour + teilgruppe['time_formated_local'].dt.minute / 60 + teilgruppe['time_formated_local'].dt.second / 3600
        
        plt.scatter(
            hours_since_midnight,
            teilgruppe['Conc'],
            color=farbe,
            marker=marker_dict.get(typ, 'o'),
            label=label_dict.get(typ, typ),
            alpha=0.7
        )

    # Title, axis labels, legend
    plt.title(f"{analyt_name}", fontsize=18)
    plt.xlabel("Hours Since Midnight", fontsize=14)
    plt.ylabel("Calculated Concentration [ppb]", fontsize=14)
    plt.legend(title="Sample Type", fontsize=14, title_fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    # X-axis formatting for hours since midnight
    ax = plt.gca()
    ax.set_xticks(range(0, 25, 1))  # Ticks every hour
    ax.set_xticklabels([f"{h}:00" for h in range(25)])  # Labels for each hour
    plt.xticks(rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    # Save
    dateiname = f"{datumsteil}_{analyt_name}_SIM_DailyPattern_linear.png"
    pfad = os.path.join(ordnername, dateiname)
    plt.savefig(pfad, bbox_inches='tight')
    plt.show()
    plt.close()





#%%
# =============================================================================
#  save dataframes as csv
# =============================================================================
csv_dateiname = f"{datumsteil}_00_df_SIM.csv"
csv_pfad = os.path.join(ordnername, csv_dateiname)
df.to_csv(csv_pfad, index=False)  # index=False to avoid saving the index as a column