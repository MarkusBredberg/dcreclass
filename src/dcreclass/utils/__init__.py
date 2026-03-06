# utils/__init__.py
# Exposes utility functions for metrics, plotting, and image processing.
# Usage: from dcreclass.utils import normalise_images, check_tensor

from .calc_tools import normalise_images, check_tensor, cluster_metrics, fold_T_axis, compute_scattering_coeffs, custom_collate
from .plotting import plot_histograms, plot_images_by_class, plot_image_grid, plot_background_histogram, plot_class_images, plot_pixel_overlaps_side_by_side