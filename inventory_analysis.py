import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def create_comparison(no_converge_run=705, converge_run=705,
                      no_converge_start=0, no_converge_end=45,
                      converge_start=0, converge_end=45,
                      show_tracking=True, show_legend=True):
    """
    Create side-by-side comparison with independent controls for each graph
    
    Parameters:
    -----------
    no_converge_run : int
        Run number for WITHOUT Converge plot (top graph)
    converge_run : int
        Run number for WITH Converge plot (bottom graph)
    no_converge_start, no_converge_end : int
        Time range for WITHOUT Converge plot
    converge_start, converge_end : int
        Time range for WITH Converge plot
    show_tracking : bool
        Whether to show tracking number circles (set to False to hide them)
    show_legend : bool
        Whether to show the legend (set to False to hide it)
    """
    
    # Load data
    df_no_converge = pd.read_csv('inventory_data_n_same.csv')
    df_converge = pd.read_csv('inventory_data_c_same.csv')
    
    # Filter for specific run and time periods (DIFFERENT for each plot)
    no_conv_run = df_no_converge[(df_no_converge['run'] == no_converge_run) & 
                                  (df_no_converge['time'] >= no_converge_start) & 
                                  (df_no_converge['time'] <= no_converge_end)].copy()
    
    conv_run = df_converge[(df_converge['run'] == converge_run) & 
                           (df_converge['time'] >= converge_start) & 
                           (df_converge['time'] <= converge_end)].copy()
    
    # Fix event ordering: SS events should happen BEFORE S events on the same day
    no_conv_run['sort_key'] = no_conv_run['event_type'].apply(lambda x: 0 if x == 'SS' else 1)
    conv_run['sort_key'] = conv_run['event_type'].apply(lambda x: 0 if x == 'SS' else 1)
    
    # Sort by time, then by sort_key (SS first)
    no_conv_run = no_conv_run.sort_values(['time', 'sort_key']).reset_index(drop=True)
    conv_run = conv_run.sort_values(['time', 'sort_key']).reset_index(drop=True)
    
    # Create figure with two subplots - SQUARE DIMENSIONS with better spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 18))
    plt.subplots_adjust(hspace=0.35, top=0.92, bottom=0.08, left=0.08, right=0.95)
    
    # Color scheme
    inventory_color = '#2E86AB'
    stockout_color = '#D62828'
    reorder_level_color = '#F77F00'
    
    # Entity type colors (consistent across request, order, and supply)
    entity_4_color = '#9B59B6'     # Purple for Entity 4 (planned demand)
    entity_5_color = '#16A085'     # Teal for Entity 5 (scheduled resupply)
    
    # Calculate max_inv from both datasets for consistent y-axis
    max_inv = max(no_conv_run['inv_level'].max(), conv_run['inv_level'].max())
    
    # ============ PLOT 1: WITHOUT CONVERGE ============
    ax1.set_title(f'WITHOUT Converge - Run {no_converge_run} (Days {no_converge_start}-{no_converge_end})', 
                  fontsize=18, fontweight='bold', pad=15, color='#C62828')
    
    # Plot base inventory level as step plot (connected)
    ax1.step(no_conv_run['time'], no_conv_run['inv_level'], 
             where='post', color=inventory_color, linewidth=3.5, 
             label='Inventory Level', zorder=3)
    
    # Plot reorder level (ignore during stockouts)
    reorder_data = no_conv_run[no_conv_run['event_type'] != 'SS'].copy()
    ax1.step(reorder_data['time'], reorder_data['reorder_level'], 
             where='post', color=reorder_level_color, linewidth=2.5, 
             linestyle='--', label='Reorder Level', zorder=2, alpha=0.7)
    
    # Fill area
    ax1.fill_between(no_conv_run['time'], 0, no_conv_run['reorder_level'], 
                     step='post', alpha=0.08, color=reorder_level_color, zorder=1)
    
    # Mark stockouts
    stockouts_no_conv = no_conv_run[no_conv_run['event_type'] == 'SS']
    for idx, (_, stockout) in enumerate(stockouts_no_conv.iterrows()):
        ax1.axvspan(stockout['time']-0.2, stockout['time']+0.2, 
                   alpha=0.15, color=stockout_color, zorder=1)
        ax1.scatter(stockout['time'], 0, color=stockout_color, s=250, marker='X', 
                   zorder=5, edgecolors='#8B0000', linewidth=2,
                   label='Stockout' if idx == 0 else '')
    
    # Create tracking system for matching request → order → supply
    visible_supplies = no_conv_run[(no_conv_run['event_type'] == 'S') & 
                                   (no_conv_run['amount_added'] > 0)].copy()
    
    supply_tracking = {}
    tracking_num = 1
    
    # Build tracking dictionary for all visible supply arrivals
    for idx, row in visible_supplies.iterrows():
        supply_key = (row['time'], row['entity_type'])
        supply_tracking[supply_key] = tracking_num
        tracking_num += 1
    
    # Show request created with entity colors
    work_created_no_conv = no_conv_run[(no_conv_run['date_work_created'] > 0) & 
                                        (no_conv_run['event_type'] == 'S')].copy()
    
    entity_4_req_labeled = False
    entity_5_req_labeled = False
    for idx, (_, row) in enumerate(work_created_no_conv.iterrows()):
        supply_key = (row['time'], row['entity_type'])
        if supply_key not in supply_tracking:
            continue
        
        if row['entity_type'] == 4:
            line_color = entity_4_color
            label = 'Invenotry Level Reorder' if not entity_4_req_labeled else ''
            entity_4_req_labeled = True
        else:
            line_color = entity_5_color
            label = 'Planned Job Reorder' if not entity_5_req_labeled else ''
            entity_5_req_labeled = True
        
        track_num = supply_tracking[supply_key]
        
        ax1.axvline(x=row['date_work_created'], color=line_color, alpha=0.5, 
                   linewidth=3, linestyle='-', zorder=2.5,
                   label=label if label else '')
        
        if show_tracking:
            ax1.text(row['date_work_created'], max_inv + 10, f"{track_num}", 
                    fontsize=10, ha='center', color=line_color, 
                    fontweight='bold', bbox=dict(boxstyle='circle,pad=0.3', 
                                                facecolor='white', edgecolor=line_color,
                                                alpha=0.9, linewidth=1.5))
    
    # Show parts ordered with entity colors
    parts_bought_no_conv = no_conv_run[(no_conv_run['date_parts_bought'] > 0) & 
                                        (no_conv_run['event_type'] == 'S')].copy()
    
    entity_4_ord_labeled = False
    entity_5_ord_labeled = False
    for idx, (_, row) in enumerate(parts_bought_no_conv.iterrows()):
        supply_key = (row['time'], row['entity_type'])
        if supply_key not in supply_tracking:
            continue
        
        if row['entity_type'] == 4:
            line_color = entity_4_color
            label =  'Planned Job Reorder'if not entity_4_ord_labeled else ''
            entity_4_ord_labeled = True
        else:
            line_color = entity_5_color
            label = 'Inventory Level Reorder' if not entity_5_ord_labeled else ''
            entity_5_ord_labeled = True
        
        track_num = supply_tracking[supply_key]
        
        ax1.axvline(x=row['date_parts_bought'], color=line_color, alpha=0.5, 
                   linewidth=3, linestyle='--', zorder=2.5,
                   label=label if label else '')
        
        if show_tracking:
            ax1.text(row['date_parts_bought'], max_inv + 10, f"{track_num}", 
                    fontsize=10, ha='center', color=line_color, 
                    fontweight='bold', bbox=dict(boxstyle='circle,pad=0.3', 
                                                facecolor='white', edgecolor=line_color,
                                                alpha=0.9, linewidth=1.5))
    
    # Overlay colored vertical lines for supply jumps
    entity_4_sup_labeled = False
    entity_5_sup_labeled = False
    for idx, row in visible_supplies.iterrows():
        if row['entity_type'] == 4:
            segment_color = entity_4_color
            label = 'Planned Job Reorder' if not entity_4_sup_labeled else ''
            entity_4_sup_labeled = True
        elif row['entity_type'] == 5:
            segment_color = entity_5_color
            label = 'Inventory Level Reorder' if not entity_5_sup_labeled else ''
            entity_5_sup_labeled = True
        else:
            continue
        
        supply_key = (row['time'], row['entity_type'])
        track_num = supply_tracking[supply_key]
        
        ax1.plot([row['time'], row['time']], 
                [row['inv_level'] - row['amount_added'], row['inv_level']], 
                color=segment_color, linewidth=5, zorder=4,
                label=label if label else '')
        
        if show_tracking:
            ax1.text(row['time'], max_inv + 10, f"{track_num}", 
                    fontsize=10, ha='center', color=segment_color, 
                    fontweight='bold', bbox=dict(boxstyle='circle,pad=0.3', 
                                                facecolor='white', edgecolor=segment_color,
                                                alpha=0.9, linewidth=1.5))
    
    ax1.set_xlabel('Time (days)', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Inventory Level (units)', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax1.set_xlim(no_converge_start, no_converge_end)
    ax1.set_ylim(-10, max_inv + (20 if show_tracking else 10))
    
    if show_legend:
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                  edgecolor='gray', fancybox=True, ncol=2)
    
    stockout_count_no_conv = len(stockouts_no_conv)
    
    # ============ PLOT 2: WITH CONVERGE ============
    ax2.set_title(f'WITH Converge - Run {converge_run} (Days {converge_start}-{converge_end})', 
                  fontsize=18, fontweight='bold', pad=15, color='#2E7D32')
    
    ax2.step(conv_run['time'], conv_run['inv_level'], 
             where='post', color=inventory_color, linewidth=3.5, 
             label='Inventory Level', zorder=3)
    
    reorder_data_conv = conv_run[conv_run['event_type'] != 'SS'].copy()
    ax2.step(reorder_data_conv['time'], reorder_data_conv['reorder_level'], 
             where='post', color=reorder_level_color, linewidth=2.5, 
             linestyle='--', label='Reorder Level', zorder=2, alpha=0.7)
    
    ax2.fill_between(conv_run['time'], 0, conv_run['reorder_level'], 
                     step='post', alpha=0.08, color=reorder_level_color, zorder=1)
    
    stockouts_conv = conv_run[conv_run['event_type'] == 'SS']
    for idx, (_, stockout) in enumerate(stockouts_conv.iterrows()):
        ax2.axvspan(stockout['time']-0.2, stockout['time']+0.2, 
                   alpha=0.15, color=stockout_color, zorder=1)
        ax2.scatter(stockout['time'], 0, color=stockout_color, s=250, marker='X', 
                   zorder=5, edgecolors='#8B0000', linewidth=2,
                   label='Stockout' if idx == 0 else '')
    
    visible_supplies_conv = conv_run[(conv_run['event_type'] == 'S') & 
                                     (conv_run['amount_added'] > 0)].copy()
    
    supply_tracking_conv = {}
    tracking_num_conv = 1
    
    for idx, row in visible_supplies_conv.iterrows():
        supply_key = (row['time'], row['entity_type'])
        supply_tracking_conv[supply_key] = tracking_num_conv
        tracking_num_conv += 1
    
    work_created_conv = conv_run[(conv_run['date_work_created'] > 0) & 
                                  (conv_run['event_type'] == 'S')].copy()
    
    entity_4_req_labeled = False
    entity_5_req_labeled = False
    for idx, (_, row) in enumerate(work_created_conv.iterrows()):
        supply_key = (row['time'], row['entity_type'])
        if supply_key not in supply_tracking_conv:
            continue
        
        if row['entity_type'] == 4:
            line_color = entity_4_color
            label = 'Planned Job Reorder' if not entity_4_req_labeled else ''
            entity_4_req_labeled = True
        else:
            line_color = entity_5_color
            label = 'Inventory Level Reorder' if not entity_5_req_labeled else ''
            entity_5_req_labeled = True
        
        track_num = supply_tracking_conv[supply_key]
        
        ax2.axvline(x=row['date_work_created'], color=line_color, alpha=0.5, 
                   linewidth=3, linestyle='-', zorder=2.5,
                   label=label if label else '')
        
        if show_tracking:
            ax2.text(row['date_work_created'], max_inv + 10, f"{track_num}", 
                    fontsize=10, ha='center', color=line_color, 
                    fontweight='bold', bbox=dict(boxstyle='circle,pad=0.3', 
                                                facecolor='white', edgecolor=line_color,
                                                alpha=0.9, linewidth=1.5))
    
    parts_bought_conv = conv_run[(conv_run['date_parts_bought'] > 0) & 
                                  (conv_run['event_type'] == 'S')].copy()
    
    entity_4_ord_labeled = False
    entity_5_ord_labeled = False
    for idx, (_, row) in enumerate(parts_bought_conv.iterrows()):
        supply_key = (row['time'], row['entity_type'])
        if supply_key not in supply_tracking_conv:
            continue
        
        if row['entity_type'] == 4:
            line_color = entity_4_color
            label = 'Planned Job Reorder' if not entity_4_ord_labeled else ''
            entity_4_ord_labeled = True
        else:
            line_color = entity_5_color
            label = 'Inventory Level Reorder' if not entity_5_ord_labeled else ''
            entity_5_ord_labeled = True
        
        track_num = supply_tracking_conv[supply_key]
        
        ax2.axvline(x=row['date_parts_bought'], color=line_color, alpha=0.5, 
                   linewidth=3, linestyle='--', zorder=2.5,
                   label=label if label else '')
        
        if show_tracking:
            ax2.text(row['date_parts_bought'], max_inv + 10, f"{track_num}", 
                    fontsize=10, ha='center', color=line_color, 
                    fontweight='bold', bbox=dict(boxstyle='circle,pad=0.3', 
                                                facecolor='white', edgecolor=line_color,
                                                alpha=0.9, linewidth=1.5))
    
    entity_4_sup_labeled = False
    entity_5_sup_labeled = False
    for idx, row in visible_supplies_conv.iterrows():
        if row['entity_type'] == 4:
            segment_color = entity_4_color
            label = 'Planned Job Reorder' if not entity_4_sup_labeled else ''
            entity_4_sup_labeled = True
        elif row['entity_type'] == 5:
            segment_color = entity_5_color
            label = 'Inventory Level Reorder' if not entity_5_sup_labeled else ''
            entity_5_sup_labeled = True
        else:
            continue
        
        supply_key = (row['time'], row['entity_type'])
        track_num = supply_tracking_conv[supply_key]
        
        ax2.plot([row['time'], row['time']], 
                [row['inv_level'] - row['amount_added'], row['inv_level']], 
                color=segment_color, linewidth=5, zorder=4,
                label=label if label else '')
        
        if show_tracking:
            ax2.text(row['time'], max_inv + 10, f"{track_num}", 
                    fontsize=10, ha='center', color=segment_color, 
                    fontweight='bold', bbox=dict(boxstyle='circle,pad=0.3', 
                                                facecolor='white', edgecolor=segment_color,
                                                alpha=0.9, linewidth=1.5))
    
    ax2.set_xlabel('Time (days)', fontsize=15, fontweight='bold')
    ax2.set_ylabel('Inventory Level (units)', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax2.set_xlim(converge_start, converge_end)
    ax2.set_ylim(-10, max_inv + (20 if show_tracking else 10))
    
    if show_legend:
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.95,
                  edgecolor='gray', fancybox=True, ncol=2)
    
    stockout_count_conv = len(stockouts_conv)
    
    reduction = stockout_count_no_conv - stockout_count_conv
    pct_reduction = (reduction / stockout_count_no_conv * 100) if stockout_count_no_conv > 0 else 0
    
    
    plt.savefig('converge_comparison_final.png', dpi=300, bbox_inches=None, pad_inches=0.1)
    print(f"\nComparison saved as 'converge_comparison_final.png'")
    print(f"\nWithout Converge (Days {no_converge_start}-{no_converge_end}): {stockout_count_no_conv} stockouts")
    print(f"With Converge (Days {converge_start}-{converge_end}): {stockout_count_conv} stockouts")
    print(f"Tracking numbers: {'ON' if show_tracking else 'OFF'}")
    print(f"Legend: {'ON' if show_legend else 'OFF'}\n")
    
    return fig

# ========== EASY CONTROLS - CHANGE THESE VALUES ==========

# Run number for top graph (WITHOUT Converge)
NO_CONVERGE_RUN = 265

# Run number for bottom graph (WITH Converge)
CONVERGE_RUN = 265

# Time range for top graph (WITHOUT Converge)
NO_CONVERGE_START = 200
NO_CONVERGE_END = 300

# Time range for bottom graph (WITH Converge)
CONVERGE_START = 200
CONVERGE_END = 300

# Show tracking number circles (True = show, False = hide)
SHOW_TRACKING = True

# Show legend (True = show, False = hide)
SHOW_LEGEND = True

# ==========================================================

fig = create_comparison(
    no_converge_run=NO_CONVERGE_RUN,
    converge_run=CONVERGE_RUN,
    no_converge_start=NO_CONVERGE_START,
    no_converge_end=NO_CONVERGE_END,
    converge_start=CONVERGE_START,
    converge_end=CONVERGE_END,
    show_tracking=SHOW_TRACKING,
    show_legend=SHOW_LEGEND
)

plt.show()
