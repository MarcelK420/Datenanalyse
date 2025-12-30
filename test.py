import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameter
# -----------------------------
target = 100.0          # Wunschdurchmesser
tol = 1.0               # ±1 mm Toleranz
spec_lower = target - tol
spec_upper = target + tol

# Xbar/R-Constants für n=3 (da du 3 Batches = 3 Werte pro Stichprobe hast)
A2 = 1.023
D3 = 0.0
D4 = 2.574

# -----------------------------
# Daten einlesen
# -----------------------------
df = pd.read_csv("daten/Process.csv", sep=";")

# Prüfen, ob Toleranzspalten vorhanden sind
# (Batch1_spec, Batch2_spec, Batch3_spec werden als 0/1 interpretiert)
spec_cols = ["Batch1_spec", "Batch2_spec", "Batch3_spec"]
batch_cols = ["Batch1", "Batch2", "Batch3"]

# -----------------------------
# Xbar- und R-Chart (über alle 3 Batches als Subgruppe)
# -----------------------------
# Subgruppen: jede Zeile ist eine Stichprobe mit n=3 Messwerten
subgroup_values = df[batch_cols].values

# Xbar je Stichprobe
xbar = subgroup_values.mean(axis=1)
# Range je Stichprobe
R = subgroup_values.max(axis=1) - subgroup_values.min(axis=1)

# Mittelwerte über alle Stichproben
Xbar_bar = xbar.mean()
R_bar = R.mean()

# Regelgrenzen Xbar-Chart
UCL_xbar = Xbar_bar + A2 * R_bar
LCL_xbar = Xbar_bar - A2 * R_bar

# Regelgrenzen R-Chart
UCL_R = D4 * R_bar
LCL_R = D3 * R_bar  # bei n=3 ist D3 = 0

# ----- Plot Xbar-Chart -----
plt.subplot(3, 1, 1)
plt.plot(xbar, marker='o', linestyle='-', label='Stichprobenmittelwerte')
plt.axhline(Xbar_bar, color='green', linestyle='--', label='CL (X̄̄)')
plt.axhline(UCL_xbar, color='red', linestyle='--', label='UCL')
plt.axhline(LCL_xbar, color='red', linestyle='--', label='LCL')
plt.title('X̄-Chart (alle Batches als Subgruppe)')
plt.xlabel('Stichprobe')
plt.ylabel('Durchmesser [mm]')
plt.legend()
plt.grid(True)

# ----- Plot R-Chart -----
plt.subplot(3, 1, 2)
plt.plot(R, marker='o', linestyle='-', label='Range je Stichprobe')
plt.axhline(R_bar, color='green', linestyle='--', label='CL (R̄)')
plt.axhline(UCL_R, color='red', linestyle='--', label='UCL')
plt.axhline(LCL_R, color='red', linestyle='--', label='LCL')
plt.title('R-Chart (alle Batches als Subgruppe)')
plt.xlabel('Stichprobe')
plt.ylabel('Range [mm]')
plt.legend()
plt.grid(True)

# -----------------------------
# P-Chart (Ausschussanteil pro Stichprobe)
# -----------------------------
# Annahme: Batch*_spec = 1 -> innerhalb ±1 mm, 0 -> außerhalb
# pro Stichprobe: 3 Teile (Batch1/2/3), also n_i = 3
spec_matrix = df[spec_cols].values
n_i = spec_matrix.shape[1]          # sollte 3 sein
p_i = 1.0 - spec_matrix.mean(axis=1)  # Anteil außerhalb Spezifikation pro Stichprobe

p_bar = p_i.mean()

# Regelgrenzen P-Chart (konstantes n_i = 3)
sigma_p = np.sqrt(p_bar * (1 - p_bar) / n_i)
UCL_p = p_bar + 3 * sigma_p
LCL_p = p_bar - 3 * sigma_p
LCL_p = max(LCL_p, 0.0)  # LCL nicht kleiner 0

plt.subplot(3, 1 ,3)
plt.plot(p_i, marker='o', linestyle='-', label='Ausschussanteil pro Stichprobe')
plt.axhline(p_bar, color='green', linestyle='--', label='CL (p̄)')
plt.axhline(UCL_p, color='red', linestyle='--', label='UCL')
plt.axhline(LCL_p, color='red', linestyle='--', label='LCL')
plt.title('P-Chart (Ausschussanteil über alle 3 Batches)')
plt.xlabel('Stichprobe')
plt.ylabel('Ausschussanteil')
plt.ylim(bottom=0)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()