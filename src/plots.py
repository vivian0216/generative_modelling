import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_robustness_results():
    # Set up the plot style
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors for each model
    colors = {
        'Baseline CNN': '#1f77b4',
        'JEM (0 steps)': '#ff7f0e', 
        'JEM (1 step)': '#2ca02c',
        'JEM (10 steps)': '#d62728',
        'JEM (25 steps)': '#9467bd'
    }
    
    # Define line styles for better distinction
    line_styles = {
        'Baseline CNN': '-',
        'JEM (0 steps)': '--', 
        'JEM (1 step)': '-.',
        'JEM (10 steps)': ':',
        'JEM (25 steps)': '-'
    }
    
    # Plot L2 results
    try:
        df_l2 = pd.read_csv('adv_results/exp_L2_results_for_plotting.csv')
        
        for column in df_l2.columns[1:]:  # Skip epsilon column
            ax1.plot(df_l2['Epsilon'], df_l2[column], 
                    color=colors[column], linestyle=line_styles[column],
                    marker='o', markersize=4, linewidth=2, label=column)
        
        ax1.set_xlabel('Epsilon', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Robustness vs PGD Attack (L2 Norm)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_ylim(0, 100)
        
    except FileNotFoundError:
        ax1.text(0.5, 0.5, 'L2 results file not found', 
                transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title('L2 Norm Results (File Not Found)', fontsize=14)
    
    # Plot L-infinity results
    try:
        df_linf = pd.read_csv('adv_results/exp_Linf_results_for_plotting.csv')
        
        for column in df_linf.columns[1:]:  # Skip epsilon column
            ax2.plot(df_linf['Epsilon'], df_linf[column], 
                    color=colors[column], linestyle=line_styles[column],
                    marker='s', markersize=4, linewidth=2, label=column)
        
        ax2.set_xlabel('Epsilon', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Robustness vs PGD Attack (L∞ Norm)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_ylim(0, 100)
        
    except FileNotFoundError:
        ax2.text(0.5, 0.5, 'L-infinity results file not found', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('L∞ Norm Results (File Not Found)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('robustness_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_stats():
    """Print some summary statistics from the results"""
    try:
        df_l2 = pd.read_csv('adv_results/exp_L2_results_for_plotting.csv')
        df_linf = pd.read_csv('adv_results/exp_Linf_results_for_plotting.csv')
        
        print("=== Summary Statistics ===")
        print(f"\nL2 Norm Results:")
        print(f"Epsilon range: {df_l2['Epsilon'].min()} - {df_l2['Epsilon'].max()}")
        
        # Find best performing model at different epsilon values
        for eps in [2.0, 5.0, 10.0]:
            if eps in df_l2['Epsilon'].values:
                row = df_l2[df_l2['Epsilon'] == eps].iloc[0]
                best_model = row[1:].idxmax()
                best_acc = row[1:].max()
                print(f"At ε={eps}: Best model is {best_model} with {best_acc}% accuracy")
        
        print(f"\nL∞ Norm Results:")
        print(f"Epsilon range: {df_linf['Epsilon'].min()} - {df_linf['Epsilon'].max()}")
        
        for eps in [2.0, 5.0, 10.0]:
            if eps in df_linf['Epsilon'].values:
                row = df_linf[df_linf['Epsilon'] == eps].iloc[0]
                best_model = row[1:].idxmax()
                best_acc = row[1:].max()
                print(f"At ε={eps}: Best model is {best_model} with {best_acc}% accuracy")
                
    except FileNotFoundError as e:
        print(f"Could not load results files: {e}")

if __name__ == "__main__":
    # Create the plots
    plot_robustness_results()
    
    # Print summary statistics
    print_summary_stats()