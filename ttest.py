import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 1. Daten laden (x.csv enthält Messwerte)
batches = pd.read_csv("daten/Process.csv", sep=';')
batch1 = list(batches['Batch1'])
batch2 = list(batches['Batch2'])

# 4. Unabhängiger t-Test (zwei Gruppen)
t_stat, p_value = stats.ttest_ind(batch1, batch2, equal_var=False)  # Welch's t-Test

print("\nT-Test Ergebnisse:")
print(f"T-Statistik: {t_stat:.4f}")
print(f"P-Wert: {p_value:.4f}")
alpha = 0.05
if p_value < alpha:
    print("Ergebnis: Signifikanter Unterschied (H0 abgelehnt)")
else:
    print("Ergebnis: Kein signifikanter Unterschied")

# 5. Visualisierung
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(batch1, alpha=0.7, label='Batch1', bins=20)
plt.hist(batch2, alpha=0.7, label='Batch2', bins=20)
plt.xlabel('Durchmesser')
plt.ylabel('Messpunkte')
plt.legend()
plt.title('Verteilungen')

plt.subplot(1, 2, 2)
box_data = [batch1, batch2]
plt.boxplot(box_data, labels=['Batch1', 'Batch2'])
plt.ylabel('Durchmesser')
plt.title('Boxplot-Vergleich')
plt.tight_layout()
plt.show()
