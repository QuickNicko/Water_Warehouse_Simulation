import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit here to rename scenarios, toggle labels, or change column
# ═══════════════════════════════════════════════════════════════════════════════

# Scenario display names — change these to rename everywhere
NAME_LEAD  = 'Lead Time'
NAME_BASE  = 'Updated Winter'
NAME_CSAME = 'Expected Average'
NAME_NSAME = 'Base'

# Column name in your CSV that holds the inventory level value
INVENTORY_COLUMN = 'inv_level'

# Toggle value labels on top of bars (True = show, False = hide)
SHOW_LABELS_PLOT1 = True   # N Same only
SHOW_LABELS_PLOT2 = True   # All 4 scenarios
SHOW_LABELS_PLOT3 = True   # 3 scenarios (no N Same)

# ═══════════════════════════════════════════════════════════════════════════════

# ── Scenario colors (matches stockout script) ─────────────────────────────────
COLORS = {
    NAME_LEAD:  '#e74c3c',
    NAME_BASE:  '#3498db',
    NAME_CSAME: '#f39c12',
    NAME_NSAME: '#9b59b6',
}

# ── Load data ─────────────────────────────────────────────────────────────────
def process_scenario(filename, scenario_name):
    df = pd.read_csv(filename)
    avg_per_run = df.groupby('run')[INVENTORY_COLUMN].mean()
    # Absolute raw max/min (integers) from full dataset before grouping
    avg_per_run.raw_max = int(df[INVENTORY_COLUMN].max())
    avg_per_run.raw_min = int(df[INVENTORY_COLUMN].min())
    # 95th percentile of per-run averages
    avg_per_run.p95 = float(np.percentile(avg_per_run, 95))
    return avg_per_run

lead_data  = process_scenario('inventory_data_lead.csv',   NAME_LEAD)
base_data  = process_scenario('inventory_data_base.csv',   NAME_BASE)
csame_data = process_scenario('inventory_data_c_same.csv', NAME_CSAME)
nsame_data = process_scenario('inventory_data_n_same.csv', NAME_NSAME)

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_stats(data):
    return {
        'mean': data.mean(),
        'std':  data.std(),
        'n':    len(data),
        'min':  data.min(),
        'max':  data.max(),
        'sem':  data.std() / np.sqrt(len(data)),
        'p95':  data.p95,
    }

def print_statistics(data, name):
    s = build_stats(data)
    print(f"\n{'='*50}")
    print(f"{name} — Average Inventory Level Statistics")
    print(f"{'='*50}")
    print(f"n (runs):        {s['n']}")
    print(f"Mean:            {s['mean']:.2f}")
    print(f"Std Dev:         {s['std']:.2f}")
    print(f"Std Error:       {s['sem']:.2f}")
    print(f"Min (avg/run):   {s['min']:.2f}")
    print(f"Max (avg/run):   {s['max']:.2f}")
    print(f"P95 (avg/run):   {s['p95']:.2f}")
    print(f"Absolute Max:    {data.raw_max}")

def plot_bar_chart(ax, scenario_dict, show_labels, title):
    """
    Draw a vertical bar chart with ±1 std dev error bars, absolute max markers,
    and 95th percentile markers.
    scenario_dict: {label: data_series}
    """
    labels = list(scenario_dict.keys())
    colors = [COLORS[l] for l in labels]
    s_list = [build_stats(scenario_dict[l]) for l in labels]
    means  = [s['mean'] for s in s_list]
    stds   = [s['std']  for s in s_list]
    maxes  = [scenario_dict[l].raw_max for l in labels]   # absolute raw max (int)
    p95s   = [scenario_dict[l].p95 for l in labels]       # 95th pct of run averages

    x = np.arange(len(labels))
    bar_width = 0.5

    ax.bar(x, means, width=bar_width, color=colors,
           edgecolor='black', linewidth=0.8, zorder=2, alpha=0.85)

    # ±1 std dev error bars
    ax.errorbar(x, means, yerr=stds, fmt='none',
                ecolor='black', elinewidth=2, capsize=8, capthick=2, zorder=3)

    # Absolute max — dotted line (:)
    for i, (mx, color) in enumerate(zip(maxes, colors)):
        ax.plot([x[i] - bar_width / 2, x[i] + bar_width / 2], [mx, mx],
                color=color, linewidth=2, linestyle=':', zorder=4)

    # 95th percentile — dashed line (--)
    for i, (p95, color) in enumerate(zip(p95s, colors)):
        ax.plot([x[i] - bar_width / 2, x[i] + bar_width / 2], [p95, p95],
                color=color, linewidth=2, linestyle='--', zorder=4)

    # Value labels above each bar (sits above the abs max line)
    if show_labels:
        for i, (mean, std, mx, p95) in enumerate(zip(means, stds, maxes, p95s)):
            label_y = mx + (max(maxes) * 0.015)
            ax.text(x[i], label_y,
                    f'Mean: {mean:.1f}\n±Std: {std:.1f}\nP95: {p95:.1f}\nMax: {mx}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Legend entries
    ax.plot([], [], color='grey', linewidth=2, linestyle=':',
            label='Absolute max observed')
    ax.plot([], [], color='grey', linewidth=2, linestyle='--',
            label='95th percentile (run avgs)')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_xlabel('Scenario',                        fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Inventory Level per Run', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)

    # Y-axis headroom — account for max line + labels
    top = max(maxes)
    ax.set_ylim(0, top * 1.25)

    return s_list

# ── Print statistics ──────────────────────────────────────────────────────────
print_statistics(lead_data,  NAME_LEAD)
print_statistics(base_data,  NAME_BASE)
print_statistics(csame_data, NAME_CSAME)
print_statistics(nsame_data, NAME_NSAME)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — N Same only
# ═══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(6, 7))

plot_bar_chart(
    ax1,
    {NAME_NSAME: nsame_data},
    SHOW_LABELS_PLOT1,
    f'Average Inventory Level: {NAME_NSAME} Scenario'
)

plt.tight_layout()
plt.savefig('inventory_avg_nsame.png')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — All four scenarios
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(12, 7))

scenarios_4 = {
    NAME_LEAD:  lead_data,
    NAME_BASE:  base_data,
    NAME_CSAME: csame_data,
    NAME_NSAME: nsame_data,
}

stats_4 = plot_bar_chart(
    ax2,
    scenarios_4,
    SHOW_LABELS_PLOT2,
    f'Average Inventory Level: {NAME_LEAD} vs {NAME_BASE} vs {NAME_CSAME} vs {NAME_NSAME}'
)

plt.tight_layout()
plt.savefig('inventory_avg_compare_4.png')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Three scenarios (no N Same)
# ═══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(10, 7))

scenarios_3 = {
    NAME_LEAD:  lead_data,
    NAME_BASE:  base_data,
    NAME_CSAME: csame_data,
}

stats_3 = plot_bar_chart(
    ax3,
    scenarios_3,
    SHOW_LABELS_PLOT3,
    f'Average Inventory Level: {NAME_LEAD} vs {NAME_BASE} vs {NAME_CSAME}'
)

plt.tight_layout()
plt.savefig('inventory_avg_compare_3.png')
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print("Summary — Average Inventory Levels (All Four Scenarios)")
print(f"{'='*85}")
print(f"{'Scenario':<15} {'Mean':<12} {'Std Dev':<12} {'Std Error':<12} {'P95':<12} {'Abs Max':<10} {'n':<6}")
print(f"{'-'*85}")
all_scenarios = {
    NAME_LEAD:  lead_data,
    NAME_BASE:  base_data,
    NAME_CSAME: csame_data,
    NAME_NSAME: nsame_data,
}
for label, data in all_scenarios.items():
    s = build_stats(data)
    print(f"{label:<15} {s['mean']:<12.2f} {s['std']:<12.2f} {s['sem']:<12.2f} {s['p95']:<12.2f} {data.raw_max:<10} {s['n']:<6}")