import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

NAME_LEAD  = 'Lead Time'
NAME_BASE  = 'Updated Winter'
NAME_CSAME = 'Expected Average'
NAME_NSAME = 'Base'

INVENTORY_COLUMN = 'inv_level'
THRESHOLD        = 250   # days above this level are counted

COLORS = {
    NAME_LEAD:  '#e74c3c',
    NAME_BASE:  '#3498db',
    NAME_CSAME: '#f39c12',
    NAME_NSAME: '#9b59b6',
}

# ═══════════════════════════════════════════════════════════════════════════════

# ── Per-run helpers ───────────────────────────────────────────────────────────
def days_above(series, threshold):
    """Total number of days (rows) above threshold."""
    return (series > threshold).sum()

def consecutive_runs_above(series, threshold):
    """
    Return a list of lengths of every consecutive run above threshold.
    Empty list if none exist.
    """
    above = (series > threshold).astype(int)
    runs, lengths = [], []
    in_run = False
    count  = 0
    for val in above:
        if val:
            in_run = True
            count += 1
        else:
            if in_run:
                runs.append(count)
            in_run = False
            count  = 0
    if in_run:
        runs.append(count)
    return runs

def process_scenario(filename):
    df = pd.read_csv(filename)
    results = []
    for run, grp in df.groupby('run'):
        inv = grp[INVENTORY_COLUMN].values
        total_days  = days_above(inv, THRESHOLD)
        cons_runs   = consecutive_runs_above(inv, THRESHOLD)
        avg_cons    = np.mean(cons_runs) if cons_runs else 0.0
        max_cons    = max(cons_runs)     if cons_runs else 0
        results.append({
            'run':       run,
            'total':     total_days,
            'avg_cons':  avg_cons,
            'max_cons':  max_cons,
        })
    return pd.DataFrame(results)

# ── Load ──────────────────────────────────────────────────────────────────────
lead_df  = process_scenario('inventory_data_lead.csv')
base_df  = process_scenario('inventory_data_base.csv')
csame_df = process_scenario('inventory_data_c_same.csv')
nsame_df = process_scenario('inventory_data_n_same.csv')

scenario_order = [NAME_LEAD, NAME_BASE, NAME_CSAME, NAME_NSAME]
data_map = {
    NAME_LEAD:  lead_df,
    NAME_BASE:  base_df,
    NAME_CSAME: csame_df,
    NAME_NSAME: nsame_df,
}

# ── Stats helper ──────────────────────────────────────────────────────────────
def col_stats(df, col):
    vals = df[col]
    return {
        'mean': vals.mean(),
        'std':  vals.std(),
        'sem':  vals.std() / np.sqrt(len(vals)),
        'max':  vals.max(),
        'p95':  np.percentile(vals, 95),
        'n':    len(vals),
    }

# ── Print statistics ──────────────────────────────────────────────────────────
METRIC_COLS  = ['total', 'avg_cons', 'max_cons']
METRIC_NAMES = {
    'total':    f'Total Days Above {THRESHOLD}',
    'avg_cons': f'Avg Consecutive Days Above {THRESHOLD}',
    'max_cons': f'Max Consecutive Days Above {THRESHOLD}',
}

for name in scenario_order:
    df = data_map[name]
    print(f"\n{'='*55}")
    print(f"{name} — Days Above {THRESHOLD} Statistics")
    print(f"{'='*55}")
    for col in METRIC_COLS:
        s = col_stats(df, col)
        fmt = '.2f' if col != 'max_cons' else '.0f'
        print(f"\n  {METRIC_NAMES[col]}")
        print(f"    Mean:      {s['mean']:{fmt}}")
        print(f"    Std Dev:   {s['std']:.2f}")
        print(f"    P95:       {s['p95']:.2f}")
        print(f"    Max:       {s['max']:{fmt}}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def draw_metric_panel(ax, col, title, ylabel, scenario_order, data_map, show_labels=True):
    """
    Single grouped bar panel for one metric across all scenarios.
    Shows mean bars, ±1 std error bars, P95 dashed line, Max dotted line.
    """
    labels = scenario_order
    colors = [COLORS[l] for l in labels]
    stats  = {l: col_stats(data_map[l], col) for l in labels}

    means = [stats[l]['mean'] for l in labels]
    stds  = [stats[l]['std']  for l in labels]
    p95s  = [stats[l]['p95']  for l in labels]
    maxes = [stats[l]['max']  for l in labels]

    x         = np.arange(len(labels))
    bar_width = 0.5

    ax.bar(x, means, width=bar_width, color=colors,
           edgecolor='black', linewidth=0.8, zorder=2, alpha=0.85)

    ax.errorbar(x, means, yerr=stds, fmt='none',
                ecolor='black', elinewidth=2, capsize=8, capthick=2, zorder=3)

    # Max — dotted
    for i, (mx, color) in enumerate(zip(maxes, colors)):
        ax.plot([x[i] - bar_width / 2, x[i] + bar_width / 2], [mx, mx],
                color=color, linewidth=2, linestyle=':', zorder=4)

    # P95 — dashed
    for i, (p95, color) in enumerate(zip(p95s, colors)):
        ax.plot([x[i] - bar_width / 2, x[i] + bar_width / 2], [p95, p95],
                color=color, linewidth=2, linestyle='--', zorder=4)

    if show_labels:
        top_val = max(maxes) if max(maxes) > 0 else 1
        for i, (mean, std, p95, mx) in enumerate(zip(means, stds, p95s, maxes)):
            label_y = mx + top_val * 0.015
            fmt = '.1f'
            ax.text(x[i], label_y,
                    f'Mean: {mean:{fmt}}\n±Std: {std:{fmt}}\nP95: {p95:{fmt}}\nMax: {mx:.0f}',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)

    top = max(maxes) if max(maxes) > 0 else 1
    ax.set_ylim(0, top * 1.30)

    return stats


# ── Figure: 3-panel layout (one panel per metric) ────────────────────────────
fig = plt.figure(figsize=(18, 7))
fig.suptitle(
    f'Inventory Days Above {THRESHOLD} — All Scenarios',
    fontsize=16, fontweight='bold', y=1.01
)

gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

draw_metric_panel(
    ax1, 'total',
    f'① Total Days Above {THRESHOLD}',
    'Days (per run)',
    scenario_order, data_map
)

draw_metric_panel(
    ax2, 'avg_cons',
    f'② Avg Consecutive Days Above {THRESHOLD}',
    'Days (avg run streak)',
    scenario_order, data_map
)

draw_metric_panel(
    ax3, 'max_cons',
    f'③ Max Consecutive Days Above {THRESHOLD}',
    'Days (longest streak per run)',
    scenario_order, data_map
)

# Shared legend (bottom center)
handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLORS[l], alpha=0.85,
                  edgecolor='black', linewidth=0.8, label=l)
    for l in scenario_order
]
handles += [
    plt.Line2D([0], [0], color='grey', linewidth=2, linestyle='--', label='95th percentile'),
    plt.Line2D([0], [0], color='grey', linewidth=2, linestyle=':',  label='Absolute max observed'),
]
fig.legend(handles=handles, loc='lower center', ncol=len(handles),
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
plt.savefig('inventory_days_above_250.png', bbox_inches='tight', dpi=150)
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"Summary — Days Above {THRESHOLD} (All Four Scenarios)")
print(f"{'='*90}")

for col, label in METRIC_NAMES.items():
    print(f"\n  {label}")
    print(f"  {'Scenario':<15} {'Mean':<10} {'Std Dev':<10} {'P95':<10} {'Max':<10} {'n':<6}")
    print(f"  {'-'*60}")
    for name in scenario_order:
        s = col_stats(data_map[name], col)
        print(f"  {name:<15} {s['mean']:<10.2f} {s['std']:<10.2f} {s['p95']:<10.2f} {s['max']:<10.0f} {s['n']:<6}")
