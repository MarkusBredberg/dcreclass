# utils/__init__.py
# Exposes utility functions for metrics, plotting, and image processing.
# Usage: from dcreclass.utils import normalise_images, check_tensor

from .calc_tools import normalise_images, check_tensor, cluster_metrics, fold_T_axis, compute_scattering_coeffs, custom_collate, round_to_1
from .plotting import plot_histograms, plot_images_by_class, plot_image_grid, plot_background_histogram, plot_class_images, plot_pixel_overlaps_side_by_side
from .fits import (ARCSEC_PER_RAD, arcsec_per_pix, fwhm_major_as, beam_cov_world,
                   beam_solid_angle_sr, kernel_from_beams, read_fits_array_header_wcs,
                   reproject_like, header_cluster_coord, robust_vmin_vmax)
from .annotation import (add_beam_patch, add_scalebar_kpc,
                          add_beam_patch_simple, add_scalebar_kpc_simple)