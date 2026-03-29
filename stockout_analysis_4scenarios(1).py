import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.patches import Patch

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit here to rename scenarios or toggle legend
# ═══════════════════════════════════════════════════════════════════════════════

# Scenario display names — change these to rename everywhere
NAME_LEAD  = 'Lead Time'
NAME_BASE  = 'Updated Winter'
NAME_CSAME = 'Expected Average'
NAME_NSAME = 'Base'

# Toggle legends on/off (True = show, False = hide)
SHOW_LEGEND_PLOT1 = True   # Base-only plot
SHOW_LEGEND_PLOT2 = True   # 4-scenario comparison
SHOW_LEGEND_PLOT3 = True   # 3-scenario comparison (no N Same)

# ═══════════════════════════════════════════════════════════════════════════════

# ── Scenario colors ───────────────────────────────────────────────────────────
COLORS = {
    NAME_LEAD:  '#e74c3c',
    NAME_BASE:  '#3498db',
    NAME_CSAME: '#f39c12',
    NAME_NSAME: '#9b59b6',
}

# ── Load data ─────────────────────────────────────────────────────────────────
def process_scenario(filename, scenario_name):
    try:
        df = pd.read_csv(filename)
        stockout_df = df[df['event_type'] == 'SS'].copy()
        stockouts_per_run = stockout_df.groupby('run').size()
        all_runs = df['run'].unique()
        stockouts_per_run = stockouts_per_run.reindex(all_runs, fill_value=0)
        return stockouts_per_run
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Using dummy data.")
        np.random.seed(42)
        seeds = {'lead': 6, 'base': 3, 'n_same': 4}
        lam = next((v for k, v in seeds.items() if k in filename), 2)
        return pd.Series(np.random.poisson(lam, 100))

lead_data  = process_scenario('inventory_data_lead.csv',   NAME_LEAD)
base_data  = process_scenario('inventory_data_base.csv',   NAME_BASE)
csame_data = process_scenario('inventory_data_c_same.csv', NAME_CSAME)
nsame_data = process_scenario('inventory_data_n_same.csv',  NAME_NSAME)

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_stats(data):
    return {
        'n':             len(data),
        'min':           data.min(),
        'median':        data.median(),
        'mean':          data.mean(),
        'max':           data.max(),
        'std':           data.std(),
        'percentile_95': np.percentile(data, 95),
    }

def print_statistics(data, name):
    s = build_stats(data)
    print(f"\n{'='*50}")
    print(f"{name} Scenario Statistics")
    print(f"{'='*50}")
    print(f"Min:      {s['min']:.2f}")
    print(f"Median:   {s['median']:.2f}")
    m = stats.mode(data, keepdims=True)
    print(f"Mode:     {m[0][0]:.2f}" if m[0].size > 0 else "Mode:     N/A")
    print(f"Mean:     {s['mean']:.2f}")
    print(f"Max:      {s['max']:.2f}")
    print(f"Std Dev:  {s['std']:.2f}")
    print(f"Count:    {s['n']}")

def make_legend_elements(label, color, s, step_style=False):
    """Return a group of legend handles for one scenario."""
    stats_text = (f'{label} Stats:\nn: {s["n"]}\nMin: {s["min"]:.2f}\n'
                  f'Median: {s["median"]:.2f}\nMean: {s["mean"]:.2f}\n'
                  f'Std Dev: {s["std"]:.2f}\nMax: {s["max"]:.2f}')
    bar_patch = (Patch(facecolor=color, alpha=0.20, edgecolor=color, linewidth=2, label=label)
                 if step_style else
                 Patch(facecolor=color, alpha=0.50, label=label))
    return [
        bar_patch,
        plt.Line2D([0], [0], color=color, linestyle='--', linewidth=2, label=f'Median: {s["median"]:.2f}'),
        plt.Line2D([0], [0], color=color, linestyle=':',  linewidth=2, label=f'95th Pct: {s["percentile_95"]:.2f}'),
        Patch(facecolor='none', edgecolor='none', label=stats_text),
        Patch(facecolor='none', edgecolor='none', label=''),  # spacer
    ]

def plot_scenarios(ax, scenario_dict, legend_order, show_legend, title):
    """
    Plot multiple scenarios on a given axes.
    scenario_dict : {label: data_series} in draw order (back to front)
    legend_order  : list of labels in desired legend order
    """
    all_stats_local = {label: build_stats(data) for label, data in scenario_dict.items()}

    data_all = np.concatenate(list(scenario_dict.values()))
    bins     = np.arange(data_all.min(), data_all.max() + 2, 1)
    bin_ctrs = (bins[:-1] + bins[1:]) / 2

    counts_dict = {}
    for label, data in scenario_dict.items():
        color    = COLORS[label]
        is_lead  = (label == NAME_LEAD)
        if is_lead:
            counts, _, _ = ax.hist(data, bins=bins, histtype='step',
                                    color=color, linewidth=2.0, label=label, zorder=4)
            ax.hist(data, bins=bins, alpha=0.20, color=color, edgecolor='none', zorder=3)
        else:
            counts, _, _ = ax.hist(data, bins=bins, alpha=0.50, label=label,
                                    color=color, edgecolor='black', linewidth=0.4, zorder=2)
        counts_dict[label] = counts

    # Error bars
    for label, counts in counts_dict.items():
        ax.errorbar(bin_ctrs, counts, yerr=np.sqrt(counts), fmt='none',
                    ecolor='black', elinewidth=1.2, capsize=2, capthick=1.2,
                    alpha=0.45, zorder=5)

    # Median and 95th percentile reference lines
    for label in legend_order:
        s     = all_stats_local[label]
        color = COLORS[label]
        ax.axvline(s['median'],        color=color, linestyle='--', linewidth=2)
        ax.axvline(s['percentile_95'], color=color, linestyle=':',  linewidth=2)

    # Legend
    if show_legend:
        legend_els = []
        for label in legend_order:
            legend_els += make_legend_elements(
                label, COLORS[label], all_stats_local[label],
                step_style=(label == NAME_LEAD)
            )
        ax.legend(handles=legend_els, loc='upper right', framealpha=0.9, fontsize=9)

    ax.set_xlabel('Number of Stockouts per Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Runs)',   fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    return all_stats_local

# ── Print statistics ──────────────────────────────────────────────────────────
print_statistics(lead_data,  NAME_LEAD)
print_statistics(base_data,  NAME_BASE)
print_statistics(csame_data, NAME_CSAME)
print_statistics(nsame_data, NAME_NSAME)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 – Base Scenario only
# ═══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(14, 7))

color_1 = '#2ecc71'
s1      = build_stats(base_data)
bins_1  = np.arange(base_data.min(), base_data.max() + 2, 1)
counts_1, _, _ = ax1.hist(base_data, bins=bins_1, alpha=0.6, color=color_1,
                           edgecolor='black', linewidth=0.5)
bin_ctrs_1 = (bins_1[:-1] + bins_1[1:]) / 2
ax1.errorbar(bin_ctrs_1, counts_1, yerr=np.sqrt(counts_1), fmt='none',
             ecolor='black', elinewidth=1.5, capsize=3, capthick=1.5, alpha=0.7)
ax1.axvline(s1['median'],        color=color_1,  linestyle='--', linewidth=2)
ax1.axvline(s1['percentile_95'], color='purple', linestyle=':',  linewidth=2)

if SHOW_LEGEND_PLOT1:
    stats_text_1 = (f"n: {s1['n']}\nMin: {s1['min']:.2f}\nMedian: {s1['median']:.2f}\n"
                    f"Mean: {s1['mean']:.2f}\nStd Dev: {s1['std']:.2f}\nMax: {s1['max']:.2f}")
    legend_elements_1 = [
        Patch(facecolor=color_1, alpha=0.6, label=NAME_BASE),
        plt.Line2D([0], [0], color=color_1,  linestyle='--', linewidth=2, label=f'Median: {s1["median"]:.2f}'),
        plt.Line2D([0], [0], color='purple', linestyle=':',  linewidth=2, label=f'95th Percentile: {s1["percentile_95"]:.2f}'),
        Patch(facecolor='none', edgecolor='none', label=stats_text_1),
    ]
    ax1.legend(handles=legend_elements_1, loc='upper right', framealpha=0.9, fontsize=10)

ax1.set_xlabel('Number of Stockouts per Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency (Number of Runs)',   fontsize=12, fontweight='bold')
ax1.set_title(f'Distribution of Annual Stockouts: {NAME_BASE} Scenario',
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('stockout_dist_base.png')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 – All four scenarios
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(16, 7))

# Draw back-to-front; Lead Time last so its outline stays on top
scenarios_4 = {
    NAME_CSAME: csame_data,
    NAME_NSAME: nsame_data,
    NAME_BASE:  base_data,
    NAME_LEAD:  lead_data,
}
legend_order_4 = [NAME_LEAD, NAME_BASE, NAME_CSAME, NAME_NSAME]
title_4 = (f'Distribution of Annual Stockouts: '
           f'{NAME_LEAD} vs {NAME_BASE} vs {NAME_CSAME} vs {NAME_NSAME}')

stats_4 = plot_scenarios(ax2, scenarios_4, legend_order_4, SHOW_LEGEND_PLOT2, title_4)
plt.tight_layout()
plt.savefig('stockout_dist_compare_4.png')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 – Three scenarios: Lead Time, Base, C Same (no N Same)
# ═══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(16, 7))

scenarios_3 = {
    NAME_CSAME: csame_data,
    NAME_BASE:  base_data,
    NAME_LEAD:  lead_data,
}
legend_order_3 = [NAME_LEAD, NAME_BASE, NAME_CSAME]
title_3 = (f'Distribution of Annual Stockouts: '
           f'{NAME_LEAD} vs {NAME_BASE} vs {NAME_CSAME}')

stats_3 = plot_scenarios(ax3, scenarios_3, legend_order_3, SHOW_LEGEND_PLOT3, title_3)
plt.tight_layout()
plt.savefig('stockout_dist_compare_3.png')
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("Summary Comparison (All Four Scenarios)")
print(f"{'='*65}")
print(f"{'Scenario':<15} {'Mean':<10} {'Median':<10} {'Std Dev':<10} {'95th Pct':<10}")
print(f"{'-'*65}")
for label in legend_order_4:
    s = stats_4[label]
    print(f"{label:<15} {s['mean']:<10.2f} {s['median']:<10.2f} "
          f"{s['std']:<10.2f} {s['percentile_95']:<10.2f}")
