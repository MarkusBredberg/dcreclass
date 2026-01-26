import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as cosmo

# Read the CSV file
df = pd.read_csv('/home/markusbredberg/Scripts/scatter_galaxies_purely_local/cluster_data.csv')

# Extract redshift column (second column named 'z')
# Remove NaN values for the histogram
redshifts = df['z'].dropna()

# Calculate statistics
mean_z = redshifts.mean()
median_z = redshifts.median()
std_z = redshifts.std()
n_clusters = len(redshifts)

# Calculate 68% confidence interval (mean ± 1 std)
lower_68 = mean_z - std_z
upper_68 = mean_z + std_z

# Create the histogram
fig, ax = plt.subplots(figsize=(12, 8))
n, bins, patches = ax.hist(redshifts, bins=30, edgecolor='black', alpha=0.7, color='#77dd77')

# Add vertical lines for mean and median
ax.axvline(mean_z, color='blue', linestyle='-', linewidth=3, label=f'Mean: {mean_z:.3f}')
ax.axvline(median_z, color='orange', linestyle='--', linewidth=3, label=f'Median: {median_z:.3f}')

# Add shaded region for 68% confidence interval
ax.axvline(lower_68, color='green', linestyle=':', linewidth=2)
ax.axvline(upper_68, color='green', linestyle=':', linewidth=2)
ax.axvspan(lower_68, upper_68, alpha=0.2, color='green', label='68% interval')

# Add labels with larger font sizes
ax.set_xlabel('z', fontsize=36)
ax.set_ylabel('Number of Clusters', fontsize=36)

# Increase tick label sizes
ax.tick_params(axis='both', labelsize=28)

# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add legend with larger font (now in upper right corner)
ax.legend(fontsize=24, loc='upper right')

# Create top x-axis for angular diameter distance
ax2 = ax.twiny()

# Get the current x-axis limits (redshift range)
z_min, z_max = ax.get_xlim()

# Create redshift values for the top axis
z_values = np.linspace(z_min, z_max, 8)

# Calculate angular diameter distances in Mpc for these redshifts
# Handle z=0 case separately to avoid division issues
da_values = []
for z in z_values:
    if z <= 0:
        da_values.append(0)
    else:
        da_values.append(cosmo.angular_diameter_distance(z).value)

# Set the top axis to match the bottom axis range
ax2.set_xlim(ax.get_xlim())

# Set tick positions and labels for angular diameter distance
ax2.set_xticks(z_values)
ax2.set_xticklabels([f'{da:.0f}' for da in da_values], fontsize=28)
ax2.set_xlabel('Angular Diameter Distance [Mpc]', fontsize=36)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure (without showing it)
plt.savefig('redshift_histogram.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

# Print summary statistics
print(f"Total clusters with redshift data: {n_clusters}")
print(f"Mean redshift: {mean_z:.3f}")
print(f"Median redshift: {median_z:.3f}")
print(f"Min redshift: {redshifts.min():.3f}")
print(f"Max redshift: {redshifts.max():.3f}")
print(f"Standard deviation: {std_z:.3f}")
print(f"68% confidence interval: [{lower_68:.3f}, {upper_68:.3f}]")
print(f"\nHistogram saved as 'redshift_histogram.png'")