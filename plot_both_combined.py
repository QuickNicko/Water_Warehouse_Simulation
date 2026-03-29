"""
Requirement 7 — Both Planned MX + Emergency Brakes (1,000 runs)
  How does running both event types change the number of stockouts per year?
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FILES = {
    'Planned Only (Fixed)':   'inventory_data_only_planned.csv',
    'Emergency Only (Fixed)': 'inventory_data_only_emergency.csv',
    'Both Combined':          'inventory_data_both.csv',
}

def get_stockouts(filepath):
    df = pd.read_csv(filepath)
    all_runs = df['run'].unique()
    return df[df['event_type'] == 'SS'].groupby('run').size().reindex(all_runs, fill_value=0)

results = {label: get_stockouts(path) for label, path in FILES.items()}

# ── Print comparison ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  COMBINED vs INDIVIDUAL SCENARIOS — 1,000 Runs Each")
print("="*65)
fmt = "{:<28} {:>8} {:>8} {:>8} {:>8} {:>8}"
print(fmt.format("Scenario", "Mean", "Median", "Std", "95th%", "Max"))
print("-"*65)
for label, ss in results.items():
    print(fmt.format(label,
                     f"{ss.mean():.2f}", f"{ss.median():.1f}",
                     f"{ss.std():.2f}",  f"{np.percentile(ss,95):.1f}",
                     f"{ss.max()}"))
print("="*65)

# How much does combining increase stockouts?
planned = results['Planned Only (Fixed)']
emergency = results['Emergency Only (Fixed)']
both = results['Both Combined']
naive_sum = planned.mean() + emergency.mean()
print(f"\n  Naive additive expectation (Planned + Emergency): {naive_sum:.2f}")
print(f"  Actual combined mean:                            {both.mean():.2f}")
print(f"  Excess above additive:                           {both.mean() - naive_sum:.2f} ({(both.mean()-naive_sum)/naive_sum*100:+.1f}%)")

t_stat, p_val = stats.ttest_ind(planned, both, equal_var=False)
print(f"\n  Welch t-test (Planned vs Both): t={t_stat:.3f}, p={p_val:.2e}")
print(f"  → {'SIGNIFICANT' if p_val < 0.05 else 'not significant'} difference at α=0.05")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle('Effect of Combining Planned MX + Emergency Breaks\n(1,000 Runs Each)',
             fontsize=13, fontweight='bold')

colors = ['steelblue', 'firebrick', 'purple']

for ax, (label, ss), color in zip(axes, results.items(), colors):
    max_bin = min(int(ss.max()) + 1, 80)  # cap display at 80 for readability
    bins = range(0, max_bin + 2)
    ax.hist(ss.clip(upper=max_bin), bins=bins, color=color,
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axvline(ss.mean(), color='gold', linestyle='--', linewidth=2,
               label=f'Mean: {ss.mean():.2f}')
    ax.axvline(np.percentile(ss, 95), color='limegreen', linestyle=':', linewidth=2,
               label=f'95th %ile: {np.percentile(ss,95):.0f}')
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlabel('Stockouts per Year', fontsize=10)
    ax.set_ylabel('Number of Runs', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=9)
    stats_txt = f"Mean={ss.mean():.2f}\nStd={ss.std():.2f}"
    ax.text(0.98, 0.95, stats_txt, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('both_combined_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: both_combined_comparison.png")

# ── Mean comparison bar chart ─────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))
labels = list(results.keys())
means  = [results[l].mean() for l in labels]
stds   = [results[l].std()  for l in labels]

bars = ax2.bar(labels, means, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
ax2.errorbar(range(len(labels)), means, yerr=stds, fmt='none',
             ecolor='black', elinewidth=1.5, capsize=6, capthick=1.5)

for bar, mean in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_ylabel('Mean Annual Stockouts', fontsize=12, fontweight='bold')
ax2.set_title('Mean Stockouts per Year by Scenario (±1 Std Dev)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, fontsize=10)
plt.tight_layout()
plt.savefig('both_mean_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: both_mean_comparison.png")
