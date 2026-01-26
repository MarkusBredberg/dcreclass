import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the tables
# Format: {model: {dataset: {metric: (mean, std)}}}

# Dataset names for the main data (Raw+RT combined)
datasets = ['Raw+RT25kpc', 'Raw+RT50kpc', 'Raw+RT100kpc', 'All versions']

# Reference datasets (RT only) - these will be shown as faint background bars
reference_datasets = ['RT25kpc', 'RT50kpc', 'RT100kpc']

# CNN data - Raw+RT combined (main data from first set of tables)
cnn_data = {
    'Raw+RT25kpc': {'Accuracy': (0.54, 0.04), 'Precision': (0.59, 0.04), 
                     'Recall': (0.54, 0.23), 'F1-score': (0.53, 0.13)},
    'Raw+RT50kpc': {'Accuracy': (0.62, 0.05), 'Precision': (0.65, 0.04), 
                     'Recall': (0.63, 0.15), 'F1-score': (0.63, 0.09)},
    'Raw+RT100kpc': {'Accuracy': (0.59, 0.05), 'Precision': (0.66, 0.05), 
                      'Recall': (0.56, 0.21), 'F1-score': (0.57, 0.12)},
    'All versions': {'Accuracy': (0.56, 0.03), 'Precision': (0.61, 0.05), 
                      'Recall': (0.58, 0.17), 'F1-score': (0.58, 0.07)}
}

# CNN reference data - RT only (from second set of tables)
cnn_reference = {
    'RT25kpc': {'Accuracy': (0.59, 0.06), 'Precision': (0.64, 0.04), 
                'Recall': (0.55, 0.16), 'F1-score': (0.58, 0.10)},
    'RT50kpc': {'Accuracy': (0.70, 0.06), 'Precision': (0.74, 0.03), 
                'Recall': (0.77, 0.14), 'F1-score': (0.74, 0.08)},
    'RT100kpc': {'Accuracy': (0.64, 0.11), 'Precision': (0.74, 0.08), 
                 'Recall': (0.58, 0.24), 'F1-score': (0.62, 0.18)}
}

# DualCSN data - Raw+RT combined (main data)
dualcsn_data = {
    'Raw+RT25kpc': {'Accuracy': (0.76, 0.04), 'Precision': (0.79, 0.03), 
                     'Recall': (0.77, 0.10), 'F1-score': (0.77, 0.05)},
    'Raw+RT50kpc': {'Accuracy': (0.73, 0.09), 'Precision': (0.78, 0.10), 
                     'Recall': (0.70, 0.16), 'F1-score': (0.73, 0.12)},
    'Raw+RT100kpc': {'Accuracy': (0.69, 0.07), 'Precision': (0.74, 0.07), 
                      'Recall': (0.70, 0.17), 'F1-score': (0.70, 0.08)},
    'All versions': {'Accuracy': (0.71, 0.04), 'Precision': (0.73, 0.03), 
                      'Recall': (0.74, 0.11), 'F1-score': (0.73, 0.06)}
}

# DualCSN reference data - RT only
dualcsn_reference = {
    'RT25kpc': {'Accuracy': (0.76, 0.04), 'Precision': (0.80, 0.04), 
                'Recall': (0.76, 0.12), 'F1-score': (0.77, 0.06)},
    'RT50kpc': {'Accuracy': (0.72, 0.14), 'Precision': (0.70, 0.24), 
                'Recall': (0.73, 0.34), 'F1-score': (0.69, 0.29)},
    'RT100kpc': {'Accuracy': (0.52, 0.13), 'Precision': (0.70, 0.11), 
                 'Recall': (0.31, 0.32), 'F1-score': (0.35, 0.27)}
}

# DualSSN data - Raw+RT combined (main data)
dualssn_data = {
    'Raw+RT25kpc': {'Accuracy': (0.80, 0.04), 'Precision': (0.81, 0.06), 
                     'Recall': (0.84, 0.12), 'F1-score': (0.81, 0.05)},
    'Raw+RT50kpc': {'Accuracy': (0.81, 0.06), 'Precision': (0.85, 0.04), 
                     'Recall': (0.80, 0.12), 'F1-score': (0.82, 0.07)},
    'Raw+RT100kpc': {'Accuracy': (0.77, 0.04), 'Precision': (0.83, 0.06), 
                      'Recall': (0.75, 0.11), 'F1-score': (0.78, 0.05)},
    'All versions': {'Accuracy': (0.76, 0.08), 'Precision': (0.77, 0.08), 
                      'Recall': (0.83, 0.10), 'F1-score': (0.79, 0.07)}
}

# DualSSN reference data - RT only
dualssn_reference = {
    'RT25kpc': {'Accuracy': (0.84, 0.02), 'Precision': (0.83, 0.02), 
                'Recall': (0.89, 0.04), 'F1-score': (0.86, 0.02)},
    'RT50kpc': {'Accuracy': (0.83, 0.02), 'Precision': (0.82, 0.02), 
                'Recall': (0.91, 0.04), 'F1-score': (0.86, 0.02)},
    'RT100kpc': {'Accuracy': (0.81, 0.02), 'Precision': (0.86, 0.05), 
                 'Recall': (0.82, 0.08), 'F1-score': (0.84, 0.03)}
}

# Color scheme - pleasant pastel colors for the three models
colors = {
    'CNN': '#77dd77',      # Pastel green
    'DualCSN': '#ff6961',  # Pastel red
    'DualSSN': '#779ecb'   # Pastel blue
}

# Metrics to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

# X positions for each dataset (0, 1, 2, 3)
x_positions = np.arange(len(datasets))

# Width of bars for grouped bar chart
bar_width = 0.25

# Offsets for each model to group bars side by side
offsets = [-bar_width, 0, bar_width]

# Create a separate figure for each metric
for metric in metrics:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # First, plot the reference bars (RT only) as faint background bars
    # These only exist for the first 3 datasets (RT25kpc, RT50kpc, RT100kpc)
    for i, (model_name, ref_data, model_color) in enumerate([
        ('CNN', cnn_reference, colors['CNN']),
        ('DualCSN', dualcsn_reference, colors['DualCSN']),
        ('DualSSN', dualssn_reference, colors['DualSSN'])
    ]):
        # Extract mean and std for this metric from reference data
        ref_means = [ref_data[f'RT{size}kpc'][metric][0] for size in [25, 50, 100]]
        ref_stds = [ref_data[f'RT{size}kpc'][metric][1] for size in [25, 50, 100]]
        
        # Plot reference bars only for first 3 positions (no 'All versions' reference)
        # Use alpha=0.2 for faint appearance and hatch pattern for distinction
        ax.bar(x_positions[:3] + offsets[i], ref_means, bar_width, 
               yerr=ref_stds, color=model_color, alpha=0.2, 
               capsize=3, edgecolor=model_color, linewidth=0.8, 
               linestyle='--', hatch='//')
    
    # Now plot the main data bars (Raw+RT combined) on top
    for i, (model_name, model_data, model_color) in enumerate([
        ('CNN', cnn_data, colors['CNN']),
        ('DualCSN', dualcsn_data, colors['DualCSN']),
        ('DualSSN', dualssn_data, colors['DualSSN'])
    ]):
        # Extract mean and std for this metric across all datasets
        means = [model_data[dataset][metric][0] for dataset in datasets]
        stds = [model_data[dataset][metric][1] for dataset in datasets]
        
        # Plot main bars with full opacity on top of reference bars
        ax.bar(x_positions + offsets[i], means, bar_width, 
               yerr=stds, label=model_name, color=model_color, 
               alpha=0.55, capsize=4, edgecolor='black', linewidth=0.7)
    
    # Formatting
    ax.set_ylabel(metric, fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(datasets, rotation=15, ha='right', fontsize=11)
    ax.set_ylim(0.25, 0.95)  # Extended range to accommodate all data
    ax.set_yticks(np.arange(0.30, 0.96, 0.05))  # Y-axis ticks every 0.05
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)  # Light grid
    ax.set_facecolor('white')
    
    # Add legend with both main models and reference indicator
    # Create custom legend handles
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors['CNN'], edgecolor='black', linewidth=0.7, 
              alpha=0.55, label='CNN'),
        Patch(facecolor=colors['DualCSN'], edgecolor='black', linewidth=0.7, 
              alpha=0.55, label='DualCSN'),
        Patch(facecolor=colors['DualSSN'], edgecolor='black', linewidth=0.7, 
              alpha=0.55, label='DualSSN'),
        Patch(facecolor='gray', edgecolor='gray', linewidth=0.8, 
              alpha=0.2, hatch='//', label='RT only (reference)')
    ]
    
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, 
             fontsize=10, framealpha=0.95, edgecolor='lightgray')
    
    # Set white background
    fig.patch.set_facecolor('white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save each metric as a separate PDF with descriptive filename
    filename = f'model_comparison_{metric.lower().replace("-", "")}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                format='pdf', facecolor='white')
    
    # Close the figure
    plt.close()
    
    print(f"Plot saved as '{filename}'")

print("\nAll plots generated successfully!")