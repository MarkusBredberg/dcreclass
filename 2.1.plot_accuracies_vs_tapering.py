import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Data extracted from the tables
# Format: {model: {dataset: (accuracy_mean, accuracy_std)}}

# Tapering sizes in kpc
tapering_sizes = [25, 50, 100]

# CNN data
cnn_normal = {
    25: (0.72, 0.07),
    50: (0.71, 0.06),
    100: (0.73, 0.12)
}

cnn_sub = {
    25: (None, None),  # No data
    50: (0.75, 0.07),
    100: (0.74, 0.04)
}

cnn_rt = {
    25: (0.59, 0.06),
    50: (0.70, 0.06),
    100: (0.64, 0.11)
}

# DualCSN data
dualcsn_normal = {
    25: (0.73, 0.11),
    50: (0.83, 0.02),
    100: (0.81, 0.02)
}

dualcsn_sub = {
    25: (None, None),  # No data
    50: (0.66, 0.09),
    100: (0.60, 0.09)
}

dualcsn_rt = {
    25: (0.76, 0.04),
    50: (0.72, 0.14),
    100: (0.52, 0.13)
}

# DualSSN data
dualssn_normal = {
    25: (0.79, 0.02),
    50: (0.82, 0.03),
    100: (0.79, 0.07)
}

dualssn_sub = {
    25: (None, None),  # No data
    50: (0.74, 0.13),
    100: (0.81, 0.05)
}

dualssn_rt = {
    25: (0.84, 0.02),
    50: (0.83, 0.02),
    100: (0.81, 0.02)
}

# Color scheme - pleasant pastel colors
colors = {
    'CNN': '#77dd77',      # Pastel green (your requested color)
    'DualCSN': '#ff6961',  # Pastel red
    'DualSSN': '#779ecb'   # Pastel blue
}

# Create the plot with larger figure size to accommodate legends
fig, ax = plt.subplots(figsize=(5, 3))

# Offset values to shift data points horizontally for clarity
# Each dataset gets a different offset
offsets = {
    'Normal': -2,  # Shift Normal points 2 units to the left
    'SUB': 0,      # Keep SUB centered
    'RT': 2        # Shift RT points 2 units to the right
}

# Plot each model-dataset combination
# CNN - using pastel green with thinner lines and offset by dataset
for data, marker, linestyle, label_suffix in [
    (cnn_normal, 'o', '-', 'Normal'),
    (cnn_sub, 's', '--', 'SUB'),
    (cnn_rt, '^', ':', 'RT')
]:
    x_vals = []
    y_vals = []
    y_errs = []
    
    for size in tapering_sizes:
        if data[size][0] is not None:
            x_vals.append(size + offsets[label_suffix])  # Apply horizontal offset by dataset
            y_vals.append(data[size][0])
            y_errs.append(data[size][1])
    
    ax.errorbar(x_vals, y_vals, yerr=y_errs, marker=marker, linestyle=linestyle,
                label=f'{label_suffix}', color=colors['CNN'], markersize=6, 
                capsize=4, linewidth=1.2, alpha=0.8)

# DualCSN - using pastel red with thinner lines and offset by dataset
for data, marker, linestyle, label_suffix in [
    (dualcsn_normal, 'o', '-', 'Normal'),
    (dualcsn_sub, 's', '--', 'SUB'),
    (dualcsn_rt, '^', ':', 'RT')
]:
    x_vals = []
    y_vals = []
    y_errs = []
    
    for size in tapering_sizes:
        if data[size][0] is not None:
            x_vals.append(size + offsets[label_suffix])  # Apply horizontal offset by dataset
            y_vals.append(data[size][0])
            y_errs.append(data[size][1])
    
    ax.errorbar(x_vals, y_vals, yerr=y_errs, marker=marker, linestyle=linestyle,
                label=f'{label_suffix}', color=colors['DualCSN'], markersize=6, 
                capsize=4, linewidth=1.2, alpha=0.8)

# DualSSN - using pastel blue with thinner lines and offset by dataset
for data, marker, linestyle, label_suffix in [
    (dualssn_normal, 'o', '-', 'Normal'),
    (dualssn_sub, 's', '--', 'SUB'),
    (dualssn_rt, '^', ':', 'RT')
]:
    x_vals = []
    y_vals = []
    y_errs = []
    
    for size in tapering_sizes:
        if data[size][0] is not None:
            x_vals.append(size + offsets[label_suffix])  # Apply horizontal offset by dataset
            y_vals.append(data[size][0])
            y_errs.append(data[size][1])
    
    ax.errorbar(x_vals, y_vals, yerr=y_errs, marker=marker, linestyle=linestyle,
                label=f'{label_suffix}', color=colors['DualSSN'], markersize=6, 
                capsize=4, linewidth=1.2, alpha=0.8)

# Formatting
ax.set_xlabel('Tapering size [kpc]', fontsize=13)
ax.set_ylabel('Accuracy', fontsize=13)
ax.set_xticks(tapering_sizes)  # Keep x-axis labels at original positions
ax.set_ylim(0.50, 0.90)

# Add more y-axis ticks for better readability
ax.set_yticks(np.arange(0.50, 0.91, 0.05))  # Y-axis ticks every 0.05

ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5) # Light grid lines for y-axis

# Set white background for clean look
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Create custom legend with two sections (Model and Dataset)
# Model legend elements with thinner lines
model_elements = [
    Line2D([0], [0], color=colors['CNN'], linewidth=1.5, label='CNN'),
    Line2D([0], [0], color=colors['DualCSN'], linewidth=1.5, label='DualCSN'),
    Line2D([0], [0], color=colors['DualSSN'], linewidth=1.5, label='DualSSN')
]

# Dataset legend elements with gray color for clarity and thinner lines
dataset_elements = [
    Line2D([0], [0], color='gray', marker='o', linestyle='-', 
           markersize=6, linewidth=1.2, label='Normal'),
    Line2D([0], [0], color='gray', marker='s', linestyle='--', 
           markersize=6, linewidth=1.2, label='SUB'),
    Line2D([0], [0], color='gray', marker='^', linestyle=':', 
           markersize=6, linewidth=1.2, label='RT')
]

# Create two legends positioned to cover the full height
# First legend (Model) at the top
legend1 = ax.legend(handles=model_elements, title='Model', 
                   loc='upper left', bbox_to_anchor=(1.01, 1.0), 
                   frameon=True, fontsize=11, title_fontsize=12,
                   framealpha=0.95, edgecolor='lightgray')
ax.add_artist(legend1)  # Add first legend back

# Second legend (Dataset) positioned lower to span remaining height
legend2 = ax.legend(handles=dataset_elements, title='Dataset',
                   loc='upper left', bbox_to_anchor=(1.01, 0.45), 
                   frameon=True, fontsize=11, title_fontsize=12,
                   framealpha=0.95, edgecolor='lightgray')

# Adjust layout to prevent legend cutoff and save as PDF
plt.tight_layout()
plt.savefig('model_accuracy_comparison.pdf', dpi=300, bbox_inches='tight', 
            format='pdf', facecolor='white')

# Close the figure without displaying it
plt.close()

print("Plot saved as 'model_accuracy_comparison.pdf'")
