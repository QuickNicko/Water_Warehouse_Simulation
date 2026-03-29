"""
Requirements 4, 5 & 6 — Emergency Brakes Only
  4) Plot 10 individual runs
  5) Statistics over 1000 runs (fixed lead time)
  6) Compare fixed vs random lead time
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FILE_FIXED  = 'inventory_data_only_emergency.csv'
FILE_RANDOM = 'inventory_data_only_emergency_random.csv'
REORDER_LEVEL = 48

df_fixed = pd.read_csv(FILE_FIXED)

# ── 4. PLOT 10 INDIVIDUAL RUNS ────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Emergency Breaks Only — 10 Individual Simulation Runs\n(Fixed Lead Time, Reorder = 48 units)',
             fontsize=14, fontweight='bold', y=0.98)

colors = plt.cm.tab10.colors

for i, run_id in enumerate(range(1, 11)):
    rdf = df_fixed[df_fixed['run'] == run_id].sort_values('time').copy()
    rdf['cum_ss'] = (rdf['event_type'] == 'SS').cumsum()

    axes[0].step(rdf['time'], rdf['inv_level'], where='post',
                 color=colors[i], alpha=0.8, linewidth=1.4, label=f'Run {run_id}')
    axes[1].step(rdf['time'], rdf['cum_ss'], where='post',
                 color=colors[i], alpha=0.8, linewidth=1.4, label=f'Run {run_id}')

axes[0].axhline(REORDER_LEVEL, color='goldenrod', linestyle='--', linewidth=1.8,
                label=f'Reorder Threshold ({REORDER_LEVEL} units)')
axes[0].axhline(0, color='red', linestyle=':', linewidth=1, alpha=0.5)

axes[0].set_ylabel('Inventory Level (Units)', fontsize=11, fontweight='bold')
axes[0].set_title('Actual Inventory Level', fontsize=11)
axes[0].legend(loc='upper right', fontsize=8, ncol=2)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].set_ylim(bottom=-5)

axes[1].set_xlabel('Day of Year', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Cumulative Stockouts', fontsize=11, fontweight='bold')
axes[1].set_title('Cumulative Inventory Stockouts', fontsize=11)
axes[1].legend(loc='upper left', fontsize=8, ncol=2)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('emergency_10_runs.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: emergency_10_runs.png")

# ── 5. STATISTICS OVER 1000 RUNS (FIXED) ─────────────────────────────────────

def get_stockouts(df):
    all_runs = df['run'].unique()
    return df[df['event_type'] == 'SS'].groupby('run').size().reindex(all_runs, fill_value=0)

fixed_ss  = get_stockouts(df_fixed)
df_random = pd.read_csv(FILE_RANDOM)
random_ss = get_stockouts(df_random)

print("\n" + "="*55)
print("  EMERGENCY BRAKES — FIXED LEAD TIME  (1,000 Runs)")
print("="*55)
print(f"  Mean stockouts/year  : {fixed_ss.mean():.3f}")
print(f"  Median               : {fixed_ss.median():.1f}")
print(f"  Std deviation        : {fixed_ss.std():.3f}")
print(f"  Min                  : {fixed_ss.min()}")
print(f"  Max                  : {fixed_ss.max()}")
print(f"  95th percentile      : {np.percentile(fixed_ss, 95):.1f}")
print(f"  Runs with 0 stockouts: {(fixed_ss == 0).sum()} ({(fixed_ss == 0).mean()*100:.1f}%)")
print("="*55)

fig2, ax = plt.subplots(figsize=(10, 5))
ax.hist(fixed_ss, bins=range(0, int(fixed_ss.max()) + 2), color='firebrick',
        edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axvline(fixed_ss.mean(), color='navy', linestyle='--', linewidth=2,
           label=f'Mean: {fixed_ss.mean():.2f}')
ax.axvline(np.percentile(fixed_ss, 95), color='orange', linestyle=':', linewidth=2,
           label=f'95th %ile: {np.percentile(fixed_ss, 95):.0f}')
ax.set_xlabel('Stockouts per Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Runs', fontsize=12, fontweight='bold')
ax.set_title('Emergency Breaks Only — Stockout Distribution (1,000 Runs, Fixed Lead Time)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
stats_box = (f"n=1000  Mean={fixed_ss.mean():.2f}\n"
             f"Std={fixed_ss.std():.2f}  Max={fixed_ss.max()}")
ax.text(0.98, 0.95, stats_box, transform=ax.transAxes, fontsize=10,
        va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('emergency_fixed_stats.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: emergency_fixed_stats.png")

# ── 6. FIXED vs RANDOM LEAD TIME COMPARISON ──────────────────────────────────

print("\n" + "="*60)
print("  EMERGENCY BRAKES — FIXED vs RANDOM LEAD TIME  (1,000 Runs Each)")
print("="*60)
fmt = "{:<28} {:>10} {:>10}"
print(fmt.format("", "Fixed LT", "Random LT"))
print("-"*60)
print(fmt.format("Mean stockouts/year",  f"{fixed_ss.mean():.3f}",  f"{random_ss.mean():.3f}"))
print(fmt.format("Median",              f"{fixed_ss.median():.1f}", f"{random_ss.median():.1f}"))
print(fmt.format("Std deviation",       f"{fixed_ss.std():.3f}",   f"{random_ss.std():.3f}"))
print(fmt.format("Min",                 f"{fixed_ss.min()}",        f"{random_ss.min()}"))
print(fmt.format("Max",                 f"{fixed_ss.max()}",        f"{random_ss.max()}"))
print(fmt.format("95th percentile",     f"{np.percentile(fixed_ss,95):.1f}", f"{np.percentile(random_ss,95):.1f}"))
print("-"*60)
delta = random_ss.mean() - fixed_ss.mean()
print(f"  Mean change with random LT: +{delta:.3f} ({delta/fixed_ss.mean()*100:+.1f}%)")
t_stat, p_val = stats.ttest_ind(fixed_ss, random_ss, equal_var=False)
print(f"  Welch t-test: t={t_stat:.3f}, p={p_val:.4f}  "
      f"({'SIGNIFICANT' if p_val < 0.05 else 'not significant'} at α=0.05)")
print("="*60)

fig3, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig3.suptitle('Emergency Breaks Only — Fixed vs Random Lead Time\n(1,000 Runs Each)',
              fontsize=13, fontweight='bold')
max_bin = max(fixed_ss.max(), random_ss.max()) + 1
bins = range(0, int(max_bin) + 2)

for ax, data, label, color in zip(axes,
                                   [fixed_ss, random_ss],
                                   ['Fixed Lead Time', 'Random Lead Time'],
                                   ['firebrick', 'darkorange']):
    ax.hist(data, bins=bins, color=color, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axvline(data.mean(), color='navy', linestyle='--', linewidth=2,
               label=f'Mean: {data.mean():.2f}')
    ax.axvline(np.percentile(data, 95), color='purple', linestyle=':', linewidth=2,
               label=f'95th %ile: {np.percentile(data,95):.0f}')
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Stockouts per Year', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    stats_txt = f"Mean={data.mean():.2f}\nStd={data.std():.2f}\nMax={data.max()}"
    ax.text(0.98, 0.95, stats_txt, transform=ax.transAxes, fontsize=10,
            va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[0].set_ylabel('Number of Runs', fontsize=11)
plt.tight_layout()
plt.savefig('emergency_fixed_vs_random.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: emergency_fixed_vs_random.png")
