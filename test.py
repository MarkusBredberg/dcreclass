import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('/users/mbredber/scratch/data/PSZ2/cluster_metadata.csv')

# Extract redshift column (second column named 'z')
# Remove NaN values for the histogram
redshifts = df['z'].dropna()

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(redshifts, bins=30, edgecolor='black', alpha=0.7, color='#77dd77')

# Add labels and title with doubled font sizes
plt.xlabel('z', fontsize=24)
plt.ylabel('Number of Clusters', fontsize=24)

# Increase tick label sizes
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Add grid for better readability
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add some statistics as text on the plot with doubled font size
mean_z = redshifts.mean()
median_z = redshifts.median()
n_clusters = len(redshifts)

stats_text = f'N = {n_clusters}\nMean z = {mean_z:.3f}\nMedian z = {median_z:.3f}'
plt.text(0.95, 0.95, stats_text, 
         transform=plt.gca().transAxes,
         fontsize=20,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('redshift_histogram.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# Print summary statistics
print(f"Total clusters with redshift data: {n_clusters}")
print(f"Mean redshift: {mean_z:.3f}")
print(f"Median redshift: {median_z:.3f}")
print(f"Min redshift: {redshifts.min():.3f}")
print(f"Max redshift: {redshifts.max():.3f}")
print(f"Standard deviation: {redshifts.std():.3f}")