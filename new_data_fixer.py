import pandas as pd

def merge_inventory_files(inventory_file='inventory_log_base.csv', stockout_file='stockouts_log_base.csv', output_file='inventory_data_base.csv'):
    """
    Merges inventory log and stockouts log into a single chronological file.
    
    Parameters:
    - inventory_file: path to inventory_log.csv
    - stockout_file: path to stockouts_log.csv
    - output_file: path for the combined output file
    """
    
    # Read the inventory log
    inv_df = pd.read_csv(inventory_file)
    inv_df.columns = inv_df.columns.str.strip()  # Clean column names
    
    # Read the stockouts log
    stock_df = pd.read_csv(stockout_file)
    stock_df.columns = stock_df.columns.str.strip()  # Clean column names
    
    # Debug: print column names to see what we're working with
    print("Inventory log columns:", inv_df.columns.tolist())
    print("Stockouts log columns:", stock_df.columns.tolist())
    print()
    
    # Map column names (handle variations)
    inv_col_map = {}
    for col in inv_df.columns:
        col_lower = col.lower()
        if 'time' in col_lower and 'time' not in inv_col_map and 'days' in col_lower:
            inv_col_map['time'] = col
        elif 'inv' in col_lower and 'level' in col_lower:
            inv_col_map['inv_level'] = col
        elif 'warehouse' in col_lower and 'demand' in col_lower:
            inv_col_map['demand'] = col
        elif 'cum' in col_lower and 'stockout' in col_lower:
            inv_col_map['cum_num'] = col
        elif 'entity' in col_lower and 'type' in col_lower:
            inv_col_map['entity_type'] = col
        elif 'amount' in col_lower and 'added' in col_lower:
            inv_col_map['amount_added'] = col
        elif 'date' in col_lower and 'planned' in col_lower and 'work' in col_lower and 'parts' in col_lower and 'bought' in col_lower:
            inv_col_map['date_parts_bought'] = col
        elif 'amount' in col_lower and 'requested' in col_lower:
            inv_col_map['amount_requested'] = col
        elif 'date' in col_lower and 'planned' in col_lower and 'work' in col_lower and 'created' in col_lower:
            inv_col_map['date_work_created'] = col
        elif 'reorder' in col_lower and 'level' in col_lower:
            inv_col_map['reorder_level'] = col
        elif 'run' in col_lower:
            inv_col_map['run'] = col
    
    stock_col_map = {}
    for col in stock_df.columns:
        col_lower = col.lower()
        if 'time' in col_lower and 'time' not in stock_col_map:
            stock_col_map['time'] = col
        elif 'inv' in col_lower and 'level' in col_lower:
            stock_col_map['inv_level'] = col
        elif 'demand' in col_lower:
            stock_col_map['demand'] = col
        elif 'cum' in col_lower:
            stock_col_map['cum_num'] = col
        elif 'entity' in col_lower and 'type' in col_lower:
            stock_col_map['entity_type'] = col
        elif 'run' in col_lower:
            stock_col_map['run'] = col
    
    print("Mapped inventory columns:", inv_col_map)
    print("Mapped stockout columns:", stock_col_map)
    print()
    
    # Process inventory data - determine event type
    # If amount_added > 0, it's a supply event (S), otherwise if demand > 0 it's a demand event (D)
    inv_df['event_type'] = 'D'  # Default to demand
    inv_df.loc[inv_df[inv_col_map['amount_added']] > 0, 'event_type'] = 'S'
    
    # For supply events, set demand to 0 (clean up the data)
    inv_df.loc[inv_df['event_type'] == 'S', inv_col_map['demand']] = 0
    
    # Select and rename columns for consistency
    inv_processed = pd.DataFrame({
        'run': inv_df[inv_col_map['run']],
        'time': inv_df[inv_col_map['time']],
        'inv_level': inv_df[inv_col_map['inv_level']],
        'demand': inv_df[inv_col_map['demand']],
        'amount_added': inv_df[inv_col_map['amount_added']],
        'cum_num': inv_df[inv_col_map['cum_num']],
        'entity_type': inv_df[inv_col_map['entity_type']],
        'date_parts_bought': inv_df[inv_col_map.get('date_parts_bought', 'date_parts_bought')].fillna(0) if 'date_parts_bought' in inv_col_map else 0,
        'amount_requested': inv_df[inv_col_map.get('amount_requested', 'amount_requested')].fillna(0) if 'amount_requested' in inv_col_map else 0,
        'date_work_created': inv_df[inv_col_map.get('date_work_created', 'date_work_created')].fillna(0) if 'date_work_created' in inv_col_map else 0,
        'reorder_level': inv_df[inv_col_map.get('reorder_level', 'reorder_level')].fillna(0) if 'reorder_level' in inv_col_map else 0,
        'event_type': inv_df['event_type']
    })
    
    # Process stockouts data
    stock_processed = pd.DataFrame({
        'run': stock_df[stock_col_map['run']],
        'time': stock_df[stock_col_map['time']],
        'inv_level': stock_df[stock_col_map['inv_level']],
        'demand': stock_df[stock_col_map['demand']],
        'amount_added': 0,  # No supply added during stockouts
        'cum_num': stock_df[stock_col_map['cum_num']],
        'entity_type': stock_df[stock_col_map['entity_type']],
        'date_parts_bought': 0,  # No parts bought for stockouts
        'amount_requested': 0,  # No request amount for stockouts
        'date_work_created': 0,  # No work created for stockouts
        'reorder_level': 0,  # No reorder level for stockouts
        'event_type': 'SS'
    })
    
    # Combine both dataframes
    combined_df = pd.concat([inv_processed, stock_processed], ignore_index=True)
    
    # Sort by run number first, then by time to maintain chronological order within each run
    combined_df = combined_df.sort_values(['run', 'time']).reset_index(drop=True)
    
    # Remove warm-up period (first 2 years = 730 days) for each run
    print("\nRemoving warm-up period (first 730 days per run)...")
    combined_df = combined_df[combined_df['time'] > 730].copy()
    
    # Subtract 730 from all time values
    combined_df['time'] = combined_df['time'] - 730
    
    # Also adjust date_parts_bought and date_work_created by subtracting 730
    combined_df['date_parts_bought'] = combined_df['date_parts_bought'].apply(lambda x: max(0, x - 730) if x > 0 else 0)
    combined_df['date_work_created'] = combined_df['date_work_created'].apply(lambda x: max(0, x - 730) if x > 0 else 0)
    
    # Remove any negative time values (shouldn't happen, but just in case)
    combined_df = combined_df[combined_df['time'] >= 0].copy()
    
    # Recalculate cumulative stockout numbers for each run
    # Reset to start from 1 for the first stockout in each run after warm-up
    print("Recalculating cumulative stockout numbers...")
    combined_df['cum_num'] = 0
    
    for run in combined_df['run'].unique():
        run_mask = combined_df['run'] == run
        stockout_mask = run_mask & (combined_df['event_type'] == 'SS')
        
        # Get indices where stockouts occur for this run
        stockout_indices = combined_df[stockout_mask].index
        
        # Assign cumulative numbers starting from 1
        combined_df.loc[stockout_indices, 'cum_num'] = range(1, len(stockout_indices) + 1)
    
    combined_df = combined_df.reset_index(drop=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    
    print(f"Successfully merged {len(inv_processed)} inventory events and {len(stock_processed)} stockout events")
    print(f"After removing warm-up period: {len(combined_df)} total events")
    print(f"Number of runs: {combined_df['run'].nunique()}")
    print(f"Output saved to: {output_file}")
    
    return combined_df

if __name__ == "__main__":
    # Run the merge
    result = merge_inventory_files()
    
    # Display first few rows as preview
    print("\nPreview of combined data:")
    print(result.head(10))
    
    # Display event type distribution
    print("\nEvent type distribution:")
    print(result['event_type'].value_counts())
    
    print("\nEvents per run:")
    print(result.groupby('run').size())