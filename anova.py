import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1. Daten laden (x.csv enthält Messwerte)
batches = pd.read_csv("daten/Process.csv", sep=";")

# 2. Mehrere Gruppen/Phasen definieren (Beispiel: 3 Phasen)
# Passe an deine tatsächlichen Phasen an (z.B. aus k_matrix.csv)
group1 = list(batches['Batch1'])
group2 = list(batches['Batch2'])
group3 = list(batches['Batch3'])

# 3. Deskriptive Statistiken
groups = [group1, group2, group3]
labels = ['Batch1', 'Batch2', 'Batch3']
for i, g in enumerate(groups):
    print(f"{labels[i]}: Mittelwert = {np.mean(g):.3f}, Std = {np.std(g):.3f}")

# 4. Einweg-ANOVA (Vergleich mehrerer Gruppen)
f_stat, p_value = stats.f_oneway(group1, group2, group3)

print("\nANOVA Ergebnisse:")
print(f"F-Statistik: {f_stat:.4f}")
print(f"P-Wert: {p_value:.4f}")
alpha = 0.05
if p_value < alpha:
    print("Ergebnis: Signifikanter Gesamtunterschied (H0 abgelehnt)")
else:
    print("Ergebnis: Kein signifikanter Gesamtunterschied")

# 5. Post-hoc Test (Tukey HSD) falls ANOVA signifikant
if p_value < alpha:
    # Daten für Tukey zusammenfassen
    data_tukey = np.concatenate(groups)
    groups_tukey = np.repeat([0, 1, 2], [len(g) for g in groups])
    tukey = pairwise_tukeyhsd(data_tukey, groups_tukey, alpha=0.05)
    print("\nTukey HSD Post-hoc Test:")
    print(tukey)

# 6. Visualisierung
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
for i, g in enumerate(groups):
    plt.hist(g, alpha=0.7, label=labels[i], bins=15)
plt.xlabel('Durchmesser')
plt.ylabel('Messpunkte')
plt.legend()
plt.title('Verteilungen der Phasen')

plt.subplot(1, 2, 2)
plt.boxplot(groups, labels=[l[:8] for l in labels])  # Labels kürzen
plt.ylabel('Durchmesser')
plt.title('Boxplot-Vergleich (ANOVA)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
