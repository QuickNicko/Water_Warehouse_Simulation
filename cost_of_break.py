import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Function to calculate cost for a single stockout based on entity_type
def calculate_stockout_cost(entity_type):
    """Calculate cost based on entity_type using triangular distribution"""
    if entity_type == 1:  # Emergency break
        return np.random.triangular(4000, 10000, 25000)
    elif entity_type == 4:  # Planned break
        return np.random.triangular(20000, 50000, 150000)
    else:
        return 0

# Function to process each scenario file
def process_scenario(filename, scenario_name):
    # Load data
    df = pd.read_csv(filename)
    
    # Filter for stockout events only
    stockout_df = df[df['event_type'] == 'SS'].copy()
    
    # Calculate cost for each stockout
    stockout_df['cost'] = stockout_df['entity_type'].apply(calculate_stockout_cost)
    
    # Sum costs per run (total annual cost per replication)
    costs_per_run = stockout_df.groupby('run')['cost'].sum()
    
    # Get all runs (including those with 0 stockouts/costs)
    all_runs = df['run'].unique()
    costs_per_run = costs_per_run.reindex(all_runs, fill_value=0)
    
    return costs_per_run, scenario_name

# Helper function to generate stats string for annotation
def get_stats_string(data):
    mean = data.mean()
    sd = data.std()
    min_val = data.min()
    max_val = data.max()
    n = len(data)
    
    stats_str = (
        f"N = {n:,}\n"
        f"Mean: ${mean:,.0f}\n"
        f"SD: ${sd:,.0f}\n"
        f"Min: ${min_val:,.0f}\n"
        f"Max: ${max_val:,.0f}"
    )
    return stats_str

# ==============================================================================
# 1. Load the two required scenarios with updated names
# ==============================================================================

# NOTE: Make sure these files exist in your directory
try:
    converge_costs, _ = process_scenario('inventory_data_lead.csv', 'With Converge')
    no_converge_costs, _ = process_scenario('inventory_data_n_same.csv', 'Base Scenario')
except FileNotFoundError:
    print("Warning: CSV files not found. Using dummy data for plotting.")
    # Create dummy data
    np.random.seed(0)
    base_costs = np.random.triangular(20000, 50000, 150000, 1000)
    no_converge_costs = pd.Series(np.random.choice(base_costs, 1000) + np.random.randint(0, 5000, 1000))
    converge_costs = pd.Series(np.random.choice(base_costs, 1000) * 0.3 + np.random.randint(0, 2000, 1000))
    converge_costs = converge_costs.reindex(no_converge_costs.index.union(converge_costs.index), fill_value=0)
    no_converge_costs = no_converge_costs.reindex(no_converge_costs.index.union(converge_costs.index), fill_value=0)


# ==============================================================================
# 2. Graph 1: Only "Without Converge" with Stats Box and Error Bars
# ==============================================================================

fig1, ax1 = plt.subplots(figsize=(12, 6))

# Define color and label
color_1 = '#3498db'
label_1 = 'Base Scenario'
dataset_1 = no_converge_costs

# Plot histogram and capture counts
bins_1 = 30
counts_1, bin_edges_1, patches_1 = ax1.hist(dataset_1, bins=bins_1, alpha=0.7, label=label_1, color=color_1, edgecolor='black', linewidth=0.5)

# Add error bars to histogram (1 standard deviation)
bin_centers_1 = (bin_edges_1[:-1] + bin_edges_1[1:]) / 2
ax1.errorbar(bin_centers_1, counts_1, yerr=np.sqrt(counts_1), fmt='none', 
             ecolor='black', elinewidth=1.5, capsize=3, capthick=1.5, alpha=0.7)

# Add median line
median_val_1 = dataset_1.median()
ax1.axvline(median_val_1, color=color_1, linestyle='--', linewidth=2)

# Add Statistics Box
stats_str_1 = get_stats_string(dataset_1)
ax1.text(0.98, 0.95, stats_str_1, 
         transform=ax1.transAxes, 
         fontsize=10, 
         verticalalignment='top', 
         horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8, edgecolor='gray'))

# Customize the plot
ax1.set_xlabel('Annual Stockout Cost ($)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency (Number of Runs)', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Annual Stockout Costs: Base Scenario', fontsize=14, fontweight='bold', pad=20)

ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)

ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.savefig('without_converge_cost_distribution_with_stats_fixed.png')
plt.show()


# ==============================================================================
# 3. Graph 2: "With Converge" vs "Without Converge" Comparison with Error Bars
# ==============================================================================

fig2, ax2 = plt.subplots(figsize=(14, 7))

# Define colors and labels for comparison
colors_2 = ['#e74c3c', '#3498db']
labels_2 = ['With Converge', 'Base Scenario']
datasets_2 = [converge_costs, no_converge_costs]

# Plot histograms and capture counts
bins_2 = 30 
all_counts = []
all_bin_edges = []
for data, color, label in zip(datasets_2, colors_2, labels_2):
    counts, bin_edges, _ = ax2.hist(data, bins=bins_2, alpha=0.6, label=label, color=color, edgecolor='black', linewidth=0.5)
    all_counts.append(counts)
    all_bin_edges.append(bin_edges)

# Add error bars to both histograms (1 standard deviation)
for counts, bin_edges, color in zip(all_counts, all_bin_edges, colors_2):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax2.errorbar(bin_centers, counts, yerr=np.sqrt(counts), fmt='none', 
                 ecolor='black', elinewidth=1.5, capsize=3, capthick=1.5, alpha=0.6)

# Add median lines
for data, color, label in zip(datasets_2, colors_2, labels_2):
    median_val = data.median()
    ax2.axvline(median_val, color=color, linestyle='--', linewidth=2)

# Add Statistics Boxes
stats_str_converge = get_stats_string(converge_costs)
ax2.text(0.98, 0.95, f"With Converge:\n{stats_str_converge}", 
         transform=ax2.transAxes, 
         fontsize=10, 
         verticalalignment='top', 
         horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='#ffe6e6', alpha=0.8, edgecolor='#e74c3c')) 

stats_str_no_converge = get_stats_string(no_converge_costs)
ax2.text(0.98, 0.65, f"Base Scenario:\n{stats_str_no_converge}", 
         transform=ax2.transAxes, 
         fontsize=10, 
         verticalalignment='top', 
         horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='#e6f0ff', alpha=0.8, edgecolor='#3498db'))

# Customize the plot
ax2.set_xlabel('Annual Stockout Cost ($)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency (Number of Runs)', fontsize=12, fontweight='bold')
ax2.set_title('Comparison of Annual Stockout Costs: With Converge vs. Base Scenario', fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper left', framealpha=0.9, fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.savefig('converge_vs_noconverge_comparison_with_stats_fixed.png')
plt.show()


# ==============================================================================
# 4. Print final summary comparison (as before)
# ==============================================================================
print(f"\n{'='*50}")
print("Summary Comparison")
print(f"{'='*50}")
print(f"{'Scenario':<20} {'Mean':<15} {'Median':<15} {'Std Dev':<15}")
print(f"{'-'*65}")
print(f"{'With Converge':<20} ${converge_costs.mean():<14,.2f} ${converge_costs.median():<14,.2f} ${converge_costs.std():<14,.2f}")
print(f"{'Without Converge':<20} ${no_converge_costs.mean():<14,.2f} ${no_converge_costs.median():<14,.2f} ${no_converge_costs.std():<14,.2f}")

# Remember to call plt.show() if you are not in an interactive environment
plt.show()