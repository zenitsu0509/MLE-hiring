import pandas as pd
import os
import sys

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_csv = os.path.join(repo_root, 'support_tickets', 'output.csv')
    airll_csv = os.path.join(repo_root, 'support_tickets', 'azur_model_5.csv')
    comparison_csv = os.path.join(repo_root, 'support_tickets', 'comparison_azure.csv')

    if not os.path.exists(output_csv):
        print(f"File not found: {output_csv}")
        return
    if not os.path.exists(airll_csv):
        print(f"File not found: {airll_csv}")
        return

    df1 = pd.read_csv(output_csv)
    df2 = pd.read_csv(airll_csv)

    print(f"Loaded {len(df1)} rows from output.csv")
    print(f"Loaded {len(df2)} rows from airll.csv")

    # Assuming both dataframes have the same rows in the same order.
    # To be safe, let's align them by the 'issue' column, or just index if they match.
    # We will do a merge on 'issue' just to be robust.
    
    # We will compare a few key classification fields
    fields_to_compare = ['status', 'request_type', 'risk_level', 'product_area', 'pii_detected']
    
    # Select columns to keep
    df1_subset = df1[['issue'] + fields_to_compare].copy()
    df2_subset = df2[['issue'] + fields_to_compare].copy()
    
    # Rename for clarity
    df1_subset.columns = ['issue'] + [f"{col}_output" for col in fields_to_compare]
    df2_subset.columns = ['issue'] + [f"{col}_airll" for col in fields_to_compare]
    
    # Merge
    merged = pd.merge(df1_subset, df2_subset, on='issue', how='outer')
    
    # Create diff columns
    for col in fields_to_compare:
        merged[f"{col}_match"] = merged[f"{col}_output"] == merged[f"{col}_airll"]
        
    # Reorder columns for easier reading
    col_order = ['issue']
    for col in fields_to_compare:
        col_order.extend([f"{col}_output", f"{col}_airll", f"{col}_match"])
        
    merged = merged[col_order]
    
    merged.to_csv(comparison_csv, index=False)
    print(f"Comparison saved to {comparison_csv}")
    
    # Print summary
    print("\nSummary of matches:")
    for col in fields_to_compare:
        match_rate = merged[f"{col}_match"].mean() * 100
        print(f"{col}: {match_rate:.2f}% match")

if __name__ == '__main__':
    main()
