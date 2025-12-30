import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# -------------------------------------------------
# 1. Daten laden
# -------------------------------------------------
x = pd.read_csv("daten/x.csv").iloc[:, 0].values  # Prozessdaten
k = pd.read_csv("daten/k_matrix.csv")

# Falls k_matrix eine Indexspalte hat, entfernen:
if not np.issubdtype(k.iloc[:, 0].dtype, np.number):
    k = k.iloc[:, 1:]

k_values = k.values  # Array der Gewichte pro Zeitstempel

# -------------------------------------------------
# 2. Änderungsrate der Gewichte berechnen
# -------------------------------------------------
# Summe der absoluten Differenzen zwischen aufeinanderfolgenden Zeilen
change = np.sum(np.abs(np.diff(k_values, axis=0)), axis=1)

# Optional glätten zur klaren Grenzfindung
change_smooth = pd.Series(change).rolling(window=10, center=True, min_periods=1).mean()

# -------------------------------------------------
# 3. Schwellwert zur Phasentrennung bestimmen
# -------------------------------------------------
# Schwellenwert empirisch / automatisch: z.B. Median + 2*Std
threshold = change_smooth.median() + 2 * change_smooth.std()

# Indizes mit starken Änderungen = Phasenwechsel
phase_change_points = np.where(change_smooth > threshold)[0]
print(f"Erkannte mögliche Phasenwechsel (Indices): {phase_change_points}")

# -------------------------------------------------
# 4. Phasen in x.csv aufteilen
# -------------------------------------------------
segment_indices = [0] + list(phase_change_points) + [len(x)]
groups = [x[segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]

print(f"Anzahl erkannter Phasen: {len(groups)}")
for i, g in enumerate(groups):
    print(f"Phase {i+1}: {len(g)} Werte, Mittelwert = {np.mean(g):.3f}")

# -------------------------------------------------
# 5. ANOVA durchführen
# -------------------------------------------------
if len(groups) >= 2:  # nur sinnvoll mit >=2 Phasen
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\nANOVA Ergebnisse:")
    print(f"F-Statistik = {f_stat:.4f}, p-Wert = {p_val:.4e}")

    alpha = 0.05
    if p_val < alpha:
        print("→ Signifikanter Unterschied zwischen mindestens zwei Phasen (H₀ abgelehnt)")

        # Tukey HSD Post-hoc Test
        data_tukey = np.concatenate(groups)
        group_labels = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
        tukey = pairwise_tukeyhsd(data_tukey, group_labels, alpha=alpha)
        print("\nTukey HSD Post-hoc Ergebnisse:")
        print(tukey)
    else:
        print("→ Kein signifikanter Unterschied (H₀ nicht abgelehnt)")

# -------------------------------------------------
# 6. Visualisierung: Phasen + Änderungsrate
# -------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(x, color='blue', alpha=0.6, label='Prozesswerte (x)')
for pc in phase_change_points:
    ax1.axvline(pc, color='red', linestyle='--', alpha=0.7)
ax1.set_xlabel("Zeit / Stichprobe")
ax1.set_ylabel("Prozesswert (x)")
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(change_smooth, color='green', label='Gewichtsänderung (k_matrix)')
ax2.axhline(threshold, color='orange', linestyle='--', label='Schwellwert')
ax2.set_ylabel("Summe abs. Änderungen")
ax2.legend(loc='upper right')

plt.title("Phasenerkennung aus k_matrix.csv und Prozesssignal x.csv")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 7. Boxplot zur ANOVA-Visualisierung
# -------------------------------------------------
plt.figure(figsize=(10, 4))
plt.boxplot(groups, labels=[f"Phase {i+1}" for i in range(len(groups))])
plt.ylabel("Prozesswerte (x)")
plt.title("Boxplot der automatisch erkannten Phasen")
plt.show()
