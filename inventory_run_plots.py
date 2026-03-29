import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Scenario display names — change to rename everywhere
NAME_LEAD  = 'Lead Time'
NAME_BASE  = 'Base'
NAME_CSAME = 'C Same'
NAME_NSAME = 'N Same'

# How many individual runs to show in the small-multiples grid
N_RUNS = 10

# Which column to use for cumulative inventory
CUMULATIVE_COLUMN = 'cum_num'

# Toggle ribbon plot and small multiples independently
SHOW_RIBBON       = True   # Mean ± std band with all runs faded behind
SHOW_SMALL_MULT   = True   # 2×5 grid of individual runs

# ═══════════════════════════════════════════════════════════════════════════════

# ── Scenario colors (matches other scripts) ───────────────────────────────────
COLORS = {
    NAME_LEAD:  '#e74c3c',
    NAME_BASE:  '#3498db',
    NAME_CSAME: '#f39c12',
    NAME_NSAME: '#9b59b6',
}

# ── File map ──────────────────────────────────────────────────────────────────
FILES = {
    NAME_LEAD:  'inventory_data_lead.csv',
    NAME_BASE:  'inventory_data_base.csv',
    NAME_CSAME: 'inventory_data_c_same.csv',
    NAME_NSAME: 'inventory_data_n_same.csv',   # fixed filename
}

# ── Load & align data ─────────────────────────────────────────────────────────
def load_scenario(filename, scenario_name):
    """
    Returns a dict: {run_id: DataFrame with columns [day, inv_level, cumulative]}
    Time is normalised to day-of-year (1–365) by ranking unique time values.
    Only the first N_RUNS runs are kept for individual-run plots.
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Warning: '{filename}' not found — using dummy data.")
        rng = np.random.default_rng(abs(hash(scenario_name)) % (2**31))
        rows = []
        for run in range(1, N_RUNS + 1):
            inv = 100
            cum = 0
            for day in range(1, 366):
                demand = rng.poisson(2)
                inv    = max(0, inv - demand + rng.poisson(0.5) * 10)
                cum   += max(0, demand)
                rows.append({'run': run, 'time': day,
                             'inv_level': inv, CUMULATIVE_COLUMN: cum})
        df = pd.DataFrame(rows)

    # Normalise time → integer day index (1-based) within each run
    df = df.sort_values(['run', 'time'])
    df['day'] = df.groupby('run')['time'].transform(
        lambda t: pd.Series(range(1, len(t) + 1), index=t.index)
    )

    runs = sorted(df['run'].unique())[:N_RUNS]
    run_dict = {}
    for r in runs:
        sub = df[df['run'] == r][['day', 'inv_level', CUMULATIVE_COLUMN]].copy()
        sub = sub.rename(columns={CUMULATIVE_COLUMN: 'cumulative'})
        run_dict[r] = sub

    return run_dict


# ── Ribbon plot ───────────────────────────────────────────────────────────────
def plot_ribbon(scenario_name, run_dict, color):
    """
    Two-panel ribbon plot:
      Top:    Actual Inventory Level
      Bottom: Cumulative Inventory (cum_num)
    All runs shown as faint lines; mean ± 1 std shown as bold line + shaded band.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f'{scenario_name} — Inventory Over Time ({N_RUNS} Runs)',
                 fontsize=15, fontweight='bold', y=1.01)

    for panel_idx, (col, ylabel) in enumerate(
        [('inv_level',  'Actual Inventory Level'),
         ('cumulative', f'Cumulative Inventory ({CUMULATIVE_COLUMN})')]):

        ax = axes[panel_idx]

        # Align all runs onto a common day index
        all_days = sorted(set(
            day for rd in run_dict.values() for day in rd['day']
        ))
        matrix = np.full((len(run_dict), len(all_days)), np.nan)
        day_idx = {d: i for i, d in enumerate(all_days)}

        for row_i, rd in enumerate(run_dict.values()):
            for _, rec in rd.iterrows():
                matrix[row_i, day_idx[rec['day']]] = rec[col]

        mean_line = np.nanmean(matrix, axis=0)
        std_line  = np.nanstd(matrix,  axis=0)

        # Individual runs — thin, faded
        for row_i in range(matrix.shape[0]):
            ax.plot(all_days, matrix[row_i], color=color,
                    alpha=0.15, linewidth=0.8, zorder=1)

        # Shaded ±1 std band
        ax.fill_between(all_days,
                        mean_line - std_line,
                        mean_line + std_line,
                        color=color, alpha=0.20, zorder=2, label='±1 Std Dev')

        # Bold mean line
        ax.plot(all_days, mean_line, color=color,
                linewidth=2.5, zorder=3, label='Mean')

        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.25, linestyle='--')
        ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

        if panel_idx == 1:
            ax.set_xlabel('Day of Year', fontsize=11, fontweight='bold')

    plt.tight_layout()
    safe = scenario_name.lower().replace(' ', '_')
    plt.savefig(f'inv_ribbon_{safe}.png', bbox_inches='tight')
    plt.show()


# ── Small multiples ───────────────────────────────────────────────────────────
def plot_small_multiples(scenario_name, run_dict, color):
    """
    2 × 5 grid — one cell per run.
    Each cell has two y-axes: inv_level (solid) and cumulative (dashed).
    """
    n_cols = 5
    n_rows = 2
    run_ids = list(run_dict.keys())

    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(
        f'{scenario_name} — Individual Runs  |  '
        f'— Actual Inventory    - - Cumulative',
        fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.45, wspace=0.35)

    for i, run_id in enumerate(run_ids[:N_RUNS]):
        row, col = divmod(i, n_cols)
        ax_left  = fig.add_subplot(gs[row, col])
        ax_right = ax_left.twinx()

        rd   = run_dict[run_id]
        days = rd['day']

        # Actual inventory — left axis, solid
        ax_left.plot(days, rd['inv_level'], color=color,
                     linewidth=1.4, linestyle='-', zorder=2)
        ax_left.set_ylabel('Inv Level', fontsize=7, color=color)
        ax_left.tick_params(axis='y', labelsize=7, labelcolor=color)
        ax_left.tick_params(axis='x', labelsize=7)

        # Cumulative — right axis, dashed, slightly darker shade
        dark = _darken(color, 0.6)
        ax_right.plot(days, rd['cumulative'], color=dark,
                      linewidth=1.4, linestyle='--', zorder=2, alpha=0.85)
        ax_right.set_ylabel('Cumul.', fontsize=7, color=dark)
        ax_right.tick_params(axis='y', labelsize=7, labelcolor=dark)

        ax_left.set_title(f'Run {run_id}', fontsize=9, fontweight='bold', pad=3)
        ax_left.set_xlabel('Day', fontsize=7)
        ax_left.grid(alpha=0.2, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe = scenario_name.lower().replace(' ', '_')
    plt.savefig(f'inv_runs_{safe}.png', bbox_inches='tight')
    plt.show()


def _darken(hex_color, factor=0.7):
    """Return a darkened version of a hex color."""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return '#{:02x}{:02x}{:02x}'.format(
        int(r * factor), int(g * factor), int(b * factor))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — generate plots for all four scenarios
# ═══════════════════════════════════════════════════════════════════════════════
scenario_order = [NAME_LEAD, NAME_BASE, NAME_CSAME, NAME_NSAME]

for name in scenario_order:
    print(f"\nProcessing: {name}")
    run_data = load_scenario(FILES[name], name)
    color    = COLORS[name]

    if SHOW_RIBBON:
        plot_ribbon(name, run_data, color)

    if SHOW_SMALL_MULT:
        plot_small_multiples(name, run_data, color)

print("\nDone. Output files saved:")
for name in scenario_order:
    safe = name.lower().replace(' ', '_')
    if SHOW_RIBBON:     print(f"  inv_ribbon_{safe}.png")
    if SHOW_SMALL_MULT: print(f"  inv_runs_{safe}.png")
