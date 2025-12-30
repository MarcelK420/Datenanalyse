import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameter
# -----------------------------
n = 1              # einstellbare Untergruppengröße (Anzahl Zeilen pro Untergruppe)
target = 100.0     # Wunschdurchmesser
tol = 1.0          # ±1 mm Toleranz (nur informativ, CL/UCL/LCL kommen aus den Daten)

# -----------------------------
# Daten einlesen
# -----------------------------
df = pd.read_csv("daten/Process.csv", sep=";")

batch_cols = ["Batch1", "Batch2", "Batch3"]
spec_cols = ["Batch1_spec", "Batch2_spec", "Batch3_spec"]

# Anzahl Stichproben
k = len(df)
g = k // n  # Anzahl vollständiger Untergruppen
if g == 0:
    raise ValueError("n ist größer als die Anzahl der Zeilen!")

# nur vollständige Gruppen verwenden
df_used = df.iloc[:g * n].copy()

# -----------------------------
# Untergruppen bilden
# -----------------------------
# Form: (g Gruppen, n Zeilen pro Gruppe, 3 Messwerte pro Zeile)
values = df_used[batch_cols].values.reshape(g, n, len(batch_cols))

# Xbar und R pro Untergruppe (über alle 3*n Teile)
group_means = values.reshape(g, -1).mean(axis=1)              # Xbar
group_ranges = values.reshape(g, -1).max(axis=1) - \
               values.reshape(g, -1).min(axis=1)              # R

# P-Chart: Spezifikationsdaten gruppieren
spec_values = df_used[spec_cols].values.reshape(g, n, len(spec_cols))
# 1 = innerhalb Spez, 0 = außerhalb
good_per_group = spec_values.sum(axis=(1, 2))                 # Anzahl innerhalb Spez
total_per_group = spec_values.shape[1] * spec_values.shape[2] # n * 3
p_i = 1.0 - (good_per_group / total_per_group)                # Ausschussanteil pro Gruppe

# -----------------------------
# CL, UCL, LCL mit ±3 Sigma
# -----------------------------
# Xbar-Chart
CL_xbar = group_means.mean()
sigma_xbar = group_means.std(ddof=1)
UCL_xbar = CL_xbar + 3 * sigma_xbar
LCL_xbar = CL_xbar - 3 * sigma_xbar

# R-Chart
CL_R = group_ranges.mean()
sigma_R = group_ranges.std(ddof=1)
UCL_R = CL_R + 3 * sigma_R
LCL_R = max(CL_R - 3 * sigma_R, 0)  # Range kann nicht < 0 sein

# P-Chart
CL_p = p_i.mean()
sigma_p = p_i.std(ddof=1)
UCL_p = CL_p + 3 * sigma_p
LCL_p = max(CL_p - 3 * sigma_p, 0.0)

# -----------------------------
# Plot: alle drei Charts in einem Fenster
# -----------------------------
x = np.arange(1, g + 1)  # Untergruppen-Index

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle(f"SPC-Charts mit Untergruppengröße n = {n}", fontsize=14)

# ---- Xbar-Chart ----
ax = axes[0]
ax.plot(x, group_means, marker='o', linestyle='-', label='X̄ pro Gruppe')
ax.axhline(CL_xbar, color='green', linestyle='--', label='CL')
ax.axhline(UCL_xbar, color='red', linestyle='--', label='UCL')
ax.axhline(LCL_xbar, color='red', linestyle='--', label='LCL')
ax.axhline(101, color='blue', linestyle='--', label='Toleranz')
ax.axhline(99, color='blue', linestyle='--')
ax.set_ylabel('Durchmesser [mm]')
ax.set_title('X̄-Chart')
ax.grid(True)
ax.legend(loc='best')

# ---- R-Chart ----
ax = axes[1]
ax.plot(x, group_ranges, marker='o', linestyle='-', label='Range pro Gruppe')
ax.axhline(CL_R, color='green', linestyle='--', label='CL')
ax.axhline(UCL_R, color='red', linestyle='--', label='UCL')
ax.axhline(LCL_R, color='red', linestyle='--', label='LCL')
ax.set_ylabel('Range [mm]')
ax.set_title('R-Chart')
ax.grid(True)
ax.legend(loc='best')

# ---- P-Chart ----
ax = axes[2]
ax.plot(x, p_i, marker='o', linestyle='-', label='Ausschussanteil pro Gruppe')
ax.axhline(CL_p, color='green', linestyle='--', label='CL')
ax.axhline(UCL_p, color='red', linestyle='--', label='UCL')
ax.axhline(LCL_p, color='red', linestyle='--', label='LCL')
ax.set_ylabel('Ausschussanteil')
ax.set_xlabel('Untergruppe')
ax.set_title('P-Chart')
ax.set_ylim(bottom=0)
ax.grid(True)
ax.legend(loc='best')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
