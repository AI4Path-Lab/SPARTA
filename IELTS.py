#conda install seaborn
#conda install scikit-image
#!/usr/bin/env python3
import os
import glob
import traceback
import warnings
from pathlib import Path
from PIL import Image
from skimage import filters, morphology, measure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import distance_transform_edt, binary_fill_holes
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from skimage import io, color, exposure
from skimage.filters import gaussian, threshold_otsu, frangi, apply_hysteresis_threshold
from skimage.feature import blob_log
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_dilation,
    binary_closing,
    disk,
    white_tophat,
    opening,
)
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops

warnings.filterwarnings("ignore", category=RuntimeWarning)
sns.set_theme(style="whitegrid")

# =========================
# USER SETTINGS
# =========================
ROOT_DIR = r"/fs/scratch/PAS2942/Alejandro/Shanaka"
GROUP_DIRS = ["Control", "PolydAdT"]
OUT_DIR = r"/fs/scratch/PAS2942/Alejandro/Shanaka/Cells"

# Detection parameters copied from calculation_map.py (8-bit RGB intensity scale).
RED_MIN_FILAMENT_SIGMA = 1
RED_MAX_FILAMENT_SIGMA = 4
RED_THRESHOLD_SCALE = 0.75
RED_MIN_OBJECT_AREA = 20
RED_MIN_SKELETON_LENGTH = 15
RED_MAX_MEAN_WIDTH = 20.0
GREEN_INTENSITY_THRESHOLD = 100
BLUE_NUCLEUS_INTENSITY_THRESHOLD = 240
NUCLEUS_EXCLUSION_DILATION_RADIUS = 5
INTERACTION_PROXIMITY_RADIUS = 4

# Cell-region estimate: not supplied in calculation_map.py, so built around
# the three detected channels. Check overlays, particularly diffuse cytoplasm.
CELL_DILATION_RADIUS = 3
CELL_CLOSING_RADIUS = 5
CELL_MIN_SIZE = 300
CELL_HOLE_AREA = 300
MIN_OBJ_SIZE = 200
HOLE_AREA = 200

N_BINS = 30
EPS = 1e-8
RNG_SEED = 7
MAX_POINTS_PER_REGION_PLOT = 12000
MAX_PIXELS_PER_IMAGE_FOR_ANOVA = 8000
PROXIMITY_RADII = (2, 5, 10)

COLORS = {
    "Nucleus": "#2563EB",
    "Actin": "#C026D3",
    "Cytoplasm": "#16A34A",
    "ActinFree": "#F97316",
    "Control": "#DC2626",
    "PolydAdT": "#2563EB",
}

# =========================
# Helpers
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(folder):
    exts = ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png", "*.PNG",
            "*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(set(files))


def infer_group(path):
    parts = Path(path).parts
    for g in GROUP_DIRS:
        if g in parts:
            return g
    return "unknown"


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def robust_rgb(img):
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        out[..., c] = normalize01(img[..., c])
    return out


def clean_mask(mask, min_size=MIN_OBJ_SIZE, hole_area=HOLE_AREA):
    mask = mask.astype(bool)
    mask = remove_small_objects(mask, min_size=min_size)
    mask = remove_small_holes(mask, area_threshold=hole_area)
    return mask.astype(bool)


def safe_mean(x):
    x = np.asarray(x)
    return float(np.mean(x)) if x.size > 0 else np.nan


def safe_std(x):
    x = np.asarray(x)
    return float(np.std(x)) if x.size > 0 else np.nan


def safe_sum(x):
    x = np.asarray(x)
    return float(np.sum(x)) if x.size > 0 else 0.0


def manders(green, mask):
    positive = green > 0
    total = np.sum(green[positive])
    if total <= 0:
        return np.nan
    return float(np.sum(green[mask]) / total)


def robust_percentile_mask(x, low=1, high=99):
    lo, hi = np.percentile(x, [low, high])
    return (x >= lo) & (x <= hi)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 10:
        return np.nan
    x = x[m]
    y = y[m]
    trim = robust_percentile_mask(x, 1, 99) & robust_percentile_mask(y, 1, 99)
    if np.sum(trim) < 10:
        return np.nan
    x = x[trim]
    y = y[trim]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(pearsonr(x, y)[0])


def downsample_values(vals, max_n=12000, seed=7):
    vals = np.asarray(vals, dtype=np.float32)
    if vals.size <= max_n:
        return vals
    rng = np.random.default_rng(seed)
    idx = rng.choice(vals.size, size=max_n, replace=False)
    return vals[idx]


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    v1 = np.var(a, ddof=1)
    v2 = np.var(b, ddof=1)
    sp = np.sqrt(((a.size - 1) * v1 + (b.size - 1) * v2) / max(a.size + b.size - 2, 1))
    if sp < 1e-12:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / sp)


def overlap_score(a, b):
    return normalize01(a) * normalize01(b)

# =========================
# I/O and plotting
# =========================
def load_image(path):
    # Matches calculation_map.py: PIL RGB conversion, uint8 per channel.
    # For 16-bit scientific TIFFs, provide an explicitly scaled 8-bit RGB
    # export rather than assuming PIL conversion preserves calibration.
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return arr.astype(np.float32) / 255.0


def save_mask(path, mask, title=None, cmap="gray"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask.astype(np.uint8), cmap=cmap)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_overlay(path, base_rgb, nucleus, actin, green):
    overlay = base_rgb.copy()
    nb = find_boundaries(nucleus, mode="outer")
    ab = find_boundaries(actin, mode="outer")
    gb = find_boundaries(green, mode="outer")
    overlay[nb] = np.array([0.15, 0.39, 0.92])
    overlay[ab] = np.array([0.75, 0.15, 0.83])
    overlay[gb] = np.array([0.09, 0.64, 0.29])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(overlay)
    ax.set_title("Mask overlay", fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_heat_overlay(path, base_rgb, heat, title, cmap="magma", alpha=0.62):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(base_rgb)
    ax.imshow(normalize01(heat), cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(g_nuc, g_act, g_cyto, g_cyto_no_actin, out_path):
    rows = []
    region_order = ["Nucleus", "Actin", "Cytoplasm", "ActinFree"]
    for name, vals in [("Nucleus", g_nuc), ("Actin", g_act), ("Cytoplasm", g_cyto), ("ActinFree", g_cyto_no_actin)]:
        vals = downsample_values(vals, max_n=MAX_POINTS_PER_REGION_PLOT, seed=RNG_SEED)
        rows.extend([{"Region": name, "GreenIntensity": float(v)} for v in vals])

    fig, ax = plt.subplots(figsize=(9, 6))
    if rows:
        df = pd.DataFrame(rows)
        sns.violinplot(data=df, x="Region", y="GreenIntensity", order=region_order,
                       palette=[COLORS[r] for r in region_order], inner="quartile", linewidth=1.4, cut=0, ax=ax)
    else:
        ax.text(0.5, 0.5, "No valid regional pixels", ha="center", va="center", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Green intensity distribution by region", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Green intensity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hexbin(x, y, xlabel, ylabel, title, out_path):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if x.size < 10:
        ax.text(0.5, 0.5, "Not enough valid pixels", ha="center", va="center", fontsize=14)
    else:
        trim = robust_percentile_mask(x, 1, 99) & robust_percentile_mask(y, 1, 99)
        x = x[trim]
        y = y[trim]
        if x.size >= 10:
            hb = ax.hexbin(x, y, gridsize=55, cmap="viridis", mincnt=1)
            cbar = fig.colorbar(hb, ax=ax)
            cbar.set_label("Pixel density")
            r = safe_pearson(x, y)
            title = f"{title}   r={r:.3f}" if np.isfinite(r) else title
        else:
            ax.text(0.5, 0.5, "Not enough valid trimmed pixels", ha="center", va="center", fontsize=14)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_radial_profile(dist_vals, signal_vals, out_path, title, ylabel):
    dist_vals = np.asarray(dist_vals, dtype=float)
    signal_vals = np.asarray(signal_vals, dtype=float)
    m = np.isfinite(dist_vals) & np.isfinite(signal_vals)
    dist_vals = dist_vals[m]
    signal_vals = signal_vals[m]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if dist_vals.size == 0 or signal_vals.size == 0:
        ax.text(0.5, 0.5, "No valid pixels", ha="center", va="center", fontsize=14)
    else:
        dmax = np.max(dist_vals)
        if dmax < 1e-8:
            ax.text(0.5, 0.5, "Distance profile unavailable", ha="center", va="center", fontsize=14)
        else:
            bins = np.linspace(0, dmax, N_BINS + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])
            prof = np.full(N_BINS, np.nan, dtype=np.float32)
            for i in range(N_BINS):
                mm = (dist_vals >= bins[i]) & (dist_vals < bins[i + 1])
                if np.any(mm):
                    prof[i] = np.mean(signal_vals[mm])
            ax.plot(centers, prof, linewidth=2.4, marker="o", markersize=3.5)
    ax.set_title(title, fontweight="bold")
    labels = {
        "Radial green profile": "Distance from nucleus (pixels)",
        "Radial green-actin colocalization profile": "Distance from actin (pixels)",
        "Radial green-actin-nucleus colocalization profile":
            "Distance from nearest actin or nucleus pixel (pixels)",
    }
    ax.set_xlabel(labels[title])
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bar(df, out_path):
    needed = ["green_nucleus_mean", "green_actin_mean", "green_cytoplasm_mean", "green_cytoplasm_no_actin_mean"]
    if any(col not in df.columns for col in needed):
        return
    plot_df = pd.DataFrame({
        "Region": ["Nucleus", "Actin", "Cytoplasm", "ActinFree"],
        "MeanGreen": [
            df["green_nucleus_mean"].mean(),
            df["green_actin_mean"].mean(),
            df["green_cytoplasm_mean"].mean(),
            df["green_cytoplasm_no_actin_mean"].mean(),
        ]
    })
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(data=plot_df, x="Region", y="MeanGreen", palette=[COLORS[r] for r in plot_df["Region"]], ax=ax)
    ax.set_title("Mean green intensity by region", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Mean green intensity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# =========================
# Detection from calculation_map.py
# =========================
def normalize_image(array: np.ndarray) -> np.ndarray:
    """Normalize an intensity array to the range 0-1."""

    array = array.astype(np.float32)
    low, high = np.percentile(array, (1.0, 99.8))

    if high <= low:
        return np.zeros_like(array, dtype=np.float32)

    return np.clip((array - low) / (high - low), 0.0, 1.0)


def float_to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert a floating-point image to an 8-bit display image."""

    maximum = float(np.max(array))

    if maximum <= 0:
        return np.zeros(array.shape, dtype=np.uint8)

    return np.clip(array / maximum * 255.0, 0, 255).astype(np.uint8)


def identify_blue_nuclei(
    blue_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    """
    Select nuclear pixels using only the raw blue-array intensity.

    Returns
    -------
    nucleus_mask:
        Boolean mask of pixels whose blue intensity is at or above the
        user-defined threshold.

    exclusion_mask:
        Dilated nucleus mask used to suppress red and green detections.

    measurements:
        Intensity and area measurements for each connected selected region.
    """

    if blue_array.ndim != 2:
        raise ValueError("The blue array must be two-dimensional.")

    if not 0 <= BLUE_NUCLEUS_INTENSITY_THRESHOLD <= 255:
        raise ValueError(
            "BLUE_NUCLEUS_INTENSITY_THRESHOLD must be between 0 and 255."
        )

    # This is the only rule used to select the nucleus.
    nucleus_mask = (
        blue_array.astype(np.float32)
        >= BLUE_NUCLEUS_INTENSITY_THRESHOLD
    )

    labeled_nuclei = measure.label(nucleus_mask, connectivity=2)
    measurements: list[dict[str, float | int]] = []

    # Labeling is used only for reporting; it does not filter the mask.
    for region in measure.regionprops(labeled_nuclei):
        object_mask = labeled_nuclei == region.label
        area = int(region.area)
        selected_intensities = blue_array[object_mask]

        measurements.append(
            {
                "nucleus_id": len(measurements) + 1,
                "area_pixels": area,
                "mean_blue_intensity": round(
                    float(np.mean(selected_intensities)),
                    3,
                ),
                "minimum_blue_intensity": int(
                    np.min(selected_intensities)
                ),
                "maximum_blue_intensity": int(
                    np.max(selected_intensities)
                ),
            }
        )

    # Dilation changes only the exclusion zone, not nucleus selection.
    exclusion_mask = morphology.binary_dilation(
        nucleus_mask,
        morphology.disk(NUCLEUS_EXCLUSION_DILATION_RADIUS),
    )

    return nucleus_mask, exclusion_mask, measurements


def identify_filaments(
    channel_array: np.ndarray,
    min_sigma: int,
    max_sigma: int,
    threshold_scale: float,
    min_object_area: int,
    min_skeleton_length: int,
    max_mean_width: float,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    """
    Detect bright line-like structures in a two-dimensional channel array.

    Returns
    -------
    filament_response:
        Floating-point Sato line-enhancement response.

    filament_mask:
        Boolean mask containing the retained filaments.

    measurements:
        Measurements for each retained filament component.
    """

    channel_normalized = normalize_image(channel_array)

    # Enhance bright tube-like or filament-like structures.
    filament_response = filters.sato(
        channel_normalized,
        sigmas=range(min_sigma, max_sigma + 1),
        black_ridges=False,
    )

    filament_response = np.nan_to_num(
        filament_response,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Remove the nuclear area before calculating the filament threshold. This
    # prevents strong nuclear staining from influencing the red cutoff.
    if exclusion_mask is not None:
        if exclusion_mask.shape != channel_array.shape:
            raise ValueError(
                "The nucleus exclusion mask and channel must have "
                "the same shape."
            )

        filament_response = filament_response.copy()
        filament_response[exclusion_mask] = 0.0

    positive_values = filament_response[filament_response > 0]

    # Return an empty result if the filter found no positive response.
    if positive_values.size == 0:
        empty_mask = np.zeros(channel_array.shape, dtype=bool)
        return filament_response, empty_mask, []

    # Values below 1.0 increase the sensitivity to weak filaments.
    otsu_threshold = filters.threshold_otsu(positive_values)
    threshold = otsu_threshold * threshold_scale

    candidate_mask = filament_response >= threshold

    if exclusion_mask is not None:
        candidate_mask &= ~exclusion_mask

    # Join small gaps in the filaments.
    candidate_mask = morphology.binary_closing(
        candidate_mask,
        morphology.disk(1),
    )

    # Closing can grow a mask by one pixel, so enforce the exclusion again.
    if exclusion_mask is not None:
        candidate_mask &= ~exclusion_mask

    # Remove very small isolated objects.
    candidate_mask = morphology.remove_small_objects(
        candidate_mask,
        min_size=min_object_area,
    )

    labeled_candidates = measure.label(candidate_mask, connectivity=2)

    filament_mask = np.zeros(candidate_mask.shape, dtype=bool)
    measurements: list[dict[str, float | int]] = []

    # Retain objects with a sufficiently long skeleton and limited width.
    for region in measure.regionprops(labeled_candidates):
        object_mask = labeled_candidates == region.label
        object_skeleton = morphology.skeletonize(object_mask)

        area = int(np.count_nonzero(object_mask))
        skeleton_length = int(np.count_nonzero(object_skeleton))
        mean_width = area / max(skeleton_length, 1)

        # Remove short objects.
        if skeleton_length < min_skeleton_length:
            continue

        # Remove broad objects, such as nuclei and large blobs.
        if mean_width > max_mean_width:
            continue

        filament_mask |= object_mask

        measurements.append(
            {
                "filament_id": len(measurements) + 1,
                "area_pixels": area,
                "skeleton_length_pixels": skeleton_length,
                "mean_width_pixels": round(mean_width, 3),
            }
        )

    return filament_response, filament_mask, measurements

def identify_green_by_intensity(
    green_array: np.ndarray,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    """
    Select green pixels using only their raw green-array intensity.

    No smoothing, line filter, morphological closing, minimum area, minimum
    length, or width filter is used to create or refine the green mask.

    Returns
    -------
    green_intensity:
        Floating-point copy of the raw green array.

    green_mask:
        Boolean mask of pixels whose green intensity is at or above the
        user-defined threshold and outside the nucleus exclusion region.

    measurements:
        Measurements for each connected selected region. Labeling and
        skeletonization are used only for reporting, not mask selection.
    """

    if green_array.ndim != 2:
        raise ValueError("The green array must be two-dimensional.")

    if not 0 <= GREEN_INTENSITY_THRESHOLD <= 255:
        raise ValueError(
            "GREEN_INTENSITY_THRESHOLD must be between 0 and 255."
        )

    green_intensity = green_array.astype(np.float32)

    # This is the only signal rule used to select green pixels.
    green_mask = green_intensity >= GREEN_INTENSITY_THRESHOLD

    if exclusion_mask is not None:
        if exclusion_mask.shape != green_array.shape:
            raise ValueError(
                "The nucleus exclusion mask and green array must have "
                "the same shape."
            )

        green_mask &= ~exclusion_mask

    labeled_regions = measure.label(green_mask, connectivity=2)
    measurements: list[dict[str, float | int]] = []

    # These calculations report properties but do not alter green_mask.
    for region in measure.regionprops(labeled_regions):
        object_mask = labeled_regions == region.label
        object_skeleton = morphology.skeletonize(object_mask)

        area = int(np.count_nonzero(object_mask))
        skeleton_length = int(np.count_nonzero(object_skeleton))
        mean_width = area / max(skeleton_length, 1)

        measurements.append(
            {
                "filament_id": len(measurements) + 1,
                "area_pixels": area,
                "skeleton_length_pixels": skeleton_length,
                "mean_width_pixels": round(mean_width, 3),
            }
        )

    return green_intensity, green_mask, measurements


def build_masks(img):
    """Reuse calculation_map detectors. Return masks expected by analysis code."""
    rgb8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    R8, G8, B8 = rgb8[..., 0], rgb8[..., 1], rgb8[..., 2]

    nucleus, exclusion, nucleus_props = identify_blue_nuclei(B8)
    red_response, actin, red_props = identify_filaments(
        R8, RED_MIN_FILAMENT_SIGMA, RED_MAX_FILAMENT_SIGMA,
        RED_THRESHOLD_SCALE, RED_MIN_OBJECT_AREA,
        RED_MIN_SKELETON_LENGTH, RED_MAX_MEAN_WIDTH,
        exclusion_mask=exclusion)
    green_raw, green, green_props = identify_green_by_intensity(
        G8, exclusion_mask=exclusion)

    # calculation_map doesn't detect a cell boundary. Build a conservative
    # inferred region from the accepted signals to allow the original region
    # analysis to run, without altering accepted red and green masks.
    seeds = nucleus | actin | green
    cell = morphology.binary_dilation(seeds, morphology.disk(CELL_DILATION_RADIUS))
    cell = morphology.binary_closing(cell, morphology.disk(CELL_CLOSING_RADIUS))
    cell = morphology.remove_small_holes(cell, area_threshold=CELL_HOLE_AREA)
    cell = morphology.remove_small_objects(cell, min_size=CELL_MIN_SIZE)
    cell |= seeds  # never discard signal pixels when building the cell estimate

    # calculation_map only defines green intensity regions, not separate
    # green puncta/filament detections. Split connected components for
    # reporting without changing the underlying green analysis mask.
    green_filament = np.zeros_like(green)
    green_puncta = np.zeros_like(green)
    green_labels = measure.label(green, connectivity=2)
    for region in measure.regionprops(green_labels):
        obj = (green_labels == region.label)
        sk_len = int(np.count_nonzero(morphology.skeletonize(obj)))
        mean_width = region.area / max(sk_len, 1)
        if sk_len >= RED_MIN_SKELETON_LENGTH and mean_width <= RED_MAX_MEAN_WIDTH:
            green_filament |= obj
        else:
            green_puncta |= obj

    cytoplasm = cell & ~nucleus
    cyto_no_actin = cytoplasm & ~actin
    green_actin_overlap = green & actin
    green_nucleus_overlap = green & nucleus
    actin_nucleus_overlap = actin & nucleus
    triple = green & actin & nucleus

    green_proximity = morphology.binary_dilation(
        green, morphology.disk(INTERACTION_PROXIMITY_RADIUS)) & ~exclusion
    interacting_red = actin & green_proximity
    red_area = int(np.count_nonzero(actin))

    return {
        "nucleus": nucleus, "nucleus_exclusion": exclusion,
        "actin": actin, "green": green,
        "cytoplasm": cytoplasm, "cyto_no_actin": cyto_no_actin,
        "cell": cell, "background": ~cell,
        "green_actin_overlap": green_actin_overlap,
        "green_nucleus_overlap": green_nucleus_overlap,
        "actin_nucleus_overlap": actin_nucleus_overlap,
        "tri_overlay_mask": triple,
        "rgb_disp": img.copy(),
        "R_s": img[..., 0], "G_s": img[..., 1], "B_s": img[..., 2],
        "G_corr": img[..., 1],
        "G_frangi": filters.sato(normalize_image(G8), sigmas=range(1, 5), black_ridges=False),
        "green_puncta": green_puncta, "green_filament": green_filament,
        "red_response": red_response,
        "interacting_red": interacting_red,
        "nucleus_count": len(nucleus_props), "red_filament_count": len(red_props),
        "green_region_count": len(green_props),
        "fraction_red_filament_area_interacting":
            (int(np.count_nonzero(interacting_red)) / red_area if red_area else 0.0),
    }

# =========================
# Statistics helpers
# =========================
def build_region_long_row(group, image_name, region_name, vals):
    vals = np.asarray(vals, dtype=float)
    return {
        "group": group,
        "image": image_name,
        "region": region_name,
        "mean_green": safe_mean(vals),
        "std_green": safe_std(vals),
        "sum_green": safe_sum(vals),
        "n_pixels": int(vals.size),
    }


def sample_pixels_for_anova(image_name, group, G, blue, actin, cell):
    idx = np.flatnonzero(cell.ravel())
    if idx.size == 0:
        return pd.DataFrame(columns=["image", "group", "green_intensity", "in_nucleus", "in_actin", "in_cell"])
    rng = np.random.default_rng(RNG_SEED)
    if idx.size > MAX_PIXELS_PER_IMAGE_FOR_ANOVA:
        idx = rng.choice(idx, size=MAX_PIXELS_PER_IMAGE_FOR_ANOVA, replace=False)
    return pd.DataFrame({
        "image": image_name,
        "group": group,
        "green_intensity": G.ravel()[idx].astype(float),
        "in_nucleus": blue.ravel()[idx].astype(int).astype(str),
        "in_actin": actin.ravel()[idx].astype(int).astype(str),
        "in_cell": cell.ravel()[idx].astype(int).astype(str),
    })


def run_three_way_anova(pixel_anova_df, out_dir):
    if pixel_anova_df.shape[0] < 20:
        return None
    try:
        model = ols("green_intensity ~ C(group) + C(image) + C(in_nucleus) * C(in_actin)", data=pixel_anova_df).fit()
        aov = anova_lm(model, typ=2).reset_index().rename(columns={"index": "term"})
        aov.to_csv(os.path.join(out_dir, "THREE_WAY_ANOVA_pixel_green_intensity.csv"), index=False)
        with open(os.path.join(out_dir, "THREE_WAY_ANOVA_pixel_green_intensity.txt"), "w") as f:
            f.write(aov.to_string(index=False) + "\n")
        return aov
    except Exception as e:
        with open(os.path.join(out_dir, "THREE_WAY_ANOVA_pixel_green_intensity_ERROR.txt"), "w") as f:
            f.write(str(e) + "\n")
        return None


def run_region_anova(region_long_df, out_dir):
    if region_long_df.shape[0] < 8:
        return None, None
    region_long_df = region_long_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_green", "region", "image", "group"])
    if region_long_df.shape[0] < 8:
        return None, None
    try:
        model = ols("mean_green ~ C(group) + C(image) + C(region)", data=region_long_df).fit()
        aov = anova_lm(model, typ=2).reset_index().rename(columns={"index": "term"})
        aov.to_csv(os.path.join(out_dir, "REGION_ANOVA_mean_green.csv"), index=False)
        tukey = pairwise_tukeyhsd(endog=region_long_df["mean_green"].values, groups=region_long_df["region"].values, alpha=0.05)
        tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tukey_df.to_csv(os.path.join(out_dir, "REGION_TUKEY_mean_green.csv"), index=False)
        with open(os.path.join(out_dir, "REGION_ANOVA_mean_green.txt"), "w") as f:
            f.write(aov.to_string(index=False) + "\n\n" + str(tukey) + "\n")
        return aov, tukey_df
    except Exception as e:
        with open(os.path.join(out_dir, "REGION_ANOVA_mean_green_ERROR.txt"), "w") as f:
            f.write(str(e) + "\n")
        return None, None


def format_tukey_strings(tukey_df):
    if tukey_df is None or tukey_df.shape[0] == 0:
        return []
    out = []
    for _, row in tukey_df.iterrows():
        p = row["p-adj"]
        try:
            p = float(p)
            out.append(f"{row['group1']} vs {row['group2']}: p={p:.3g}")
        except Exception:
            out.append(f"{row['group1']} vs {row['group2']}: p={p}")
    return out[:4]


def plot_manders_boxplot(df, out_path, region_pvals=None):
    cols = ["manders_green_nucleus", "manders_green_actin", "manders_green_cytoplasm", "manders_green_actin_free"]
    present_cols = [c for c in cols if c in df.columns]
    if not present_cols:
        return
    box_df = df[present_cols].copy().rename(columns={
        "manders_green_nucleus": "Nucleus",
        "manders_green_actin": "Actin",
        "manders_green_cytoplasm": "Cytoplasm",
        "manders_green_actin_free": "ActinFree",
    })
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=box_df[["Nucleus", "Actin", "Cytoplasm", "ActinFree"]],
                palette=[COLORS["Nucleus"], COLORS["Actin"], COLORS["Cytoplasm"], COLORS["ActinFree"]], ax=ax)
    ax.set_ylabel("Manders overlap (green)")
    ax.set_title("Green overlap with compartments", fontweight="bold")
    if region_pvals:
        fig.text(0.5, 0.01, " | ".join(region_pvals), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# =========================
# Group comparison
# =========================
def compare_groups(summary_df, out_dir):
    ensure_dir(out_dir)
    numeric_cols = [c for c in summary_df.columns if pd.api.types.is_numeric_dtype(summary_df[c]) and c not in {"file_index"}]
    results = []
    g1, g2 = GROUP_DIRS
    for col in numeric_cols:
        a = summary_df.loc[summary_df["group"] == g1, col].dropna().values
        b = summary_df.loc[summary_df["group"] == g2, col].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        try:
            wt = ttest_ind(a, b, equal_var=False, nan_policy="omit")
            mu = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            continue
        results.append({
            "metric": col,
            f"{g1}_n": len(a),
            f"{g2}_n": len(b),
            f"{g1}_mean": float(np.mean(a)),
            f"{g2}_mean": float(np.mean(b)),
            f"{g1}_std": float(np.std(a, ddof=1)) if len(a) > 1 else np.nan,
            f"{g2}_std": float(np.std(b, ddof=1)) if len(b) > 1 else np.nan,
            "delta_mean": float(np.mean(a) - np.mean(b)),
            "welch_t_p": float(wt.pvalue),
            "mannwhitney_p": float(mu.pvalue),
            "cohen_d": cohen_d(a, b),
        })
    stats_df = pd.DataFrame(results)
    if stats_df.shape[0] > 0:
        stats_df["welch_t_fdr"] = multipletests(stats_df["welch_t_p"].values, method="fdr_bh")[1]
        stats_df["mannwhitney_fdr"] = multipletests(stats_df["mannwhitney_p"].values, method="fdr_bh")[1]
        stats_df = stats_df.sort_values(["welch_t_fdr", "mannwhitney_fdr", "metric"]).reset_index(drop=True)
    stats_df.to_csv(os.path.join(out_dir, "group_comparison_stats.csv"), index=False)
    return stats_df


def build_group_summary(summary_df, out_path):
    numeric_cols = [c for c in summary_df.columns if pd.api.types.is_numeric_dtype(summary_df[c])]
    rows = []
    for group, sub in summary_df.groupby("group"):
        row = {"group": group, "n_images": sub.shape[0]}
        for c in numeric_cols:
            row[f"{c}_mean"] = safe_mean(sub[c].dropna().values)
            row[f"{c}_std"] = safe_std(sub[c].dropna().values)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def make_group_plots(summary_df, stats_df, out_dir):
    ensure_dir(out_dir)
    priority = [
        "green_actin_mean", "green_cytoplasm_mean", "manders_green_actin",
        "mean_green_actin_coloc_score", "mean_green_to_actin_distance",
        "fraction_green_within_5px_actin", "puncta_count", "green_filament_pixels"
    ]
    metrics = [m for m in priority if m in summary_df.columns]
    titles = {
        "green_actin_mean": "Mean green intensity within actin",
        "green_cytoplasm_mean": "Mean green intensity within cytoplasm",
        "manders_green_actin": "Green–actin overlap coefficient",
        "mean_green_actin_coloc_score": "Green–actin colocalization score",
        "mean_green_to_actin_distance": "Mean green-to-actin distance",
        "fraction_green_within_5px_actin": "Green signal within 5 pixels of actin",
        "puncta_count": "Number of green puncta",
        "green_filament_pixels": "Green filament area",
    }

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))

        sns.stripplot(
            data=summary_df, x=metric, y="group",
            order=GROUP_DIRS, hue="group", hue_order=GROUP_DIRS,
            palette=["#F4A582", "#70C1E4"],
            jitter=0.25, size=8, alpha=0.7,
            legend=False, ax=ax
        )

        sns.pointplot(
            data=summary_df, x=metric, y="group",
            order=GROUP_DIRS, orient="h",
            errorbar=("ci", 95), n_boot=10000, seed=42,
            linestyle="none", color="black",
            markers="o", markersize=5, capsize=0.1,
            err_kws={"linewidth": 1.5}, zorder=10, ax=ax
        )

        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(titles.get(metric, metric))
        ax.set_xlabel("Mean and 95% confidence interval")
        ax.set_ylabel("")
        ax.grid(color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)

        for i, group in enumerate(GROUP_DIRS):
            n = summary_df.loc[
                summary_df["group"] == group, metric
            ].count()
            ax.text(
                0.97, i - 0.35, f"n = {n}",
                transform=ax.get_yaxis_transform(),
                ha="right", fontsize=10
            )

        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"{metric}_by_group.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    if stats_df is not None and stats_df.shape[0] > 0:
        top = stats_df.head(min(15, stats_df.shape[0])).copy()
        top = top.iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * top.shape[0] + 1.5)))
        ax.barh(top["metric"], -np.log10(top["welch_t_fdr"].clip(lower=1e-300)))
        ax.set_xlabel("-log10 FDR")
        ax.set_title("Top group differences", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "TOP_group_differences.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

# =========================
# Per-image analysis
# =========================
def analyze_image(img_path, file_index, out_root):
    group = infer_group(img_path)
    name = Path(img_path).stem
    outdir = os.path.join(out_root, group, name)
    ensure_dir(outdir)

    img = load_image(img_path)
    masks = build_masks(img)

    blue = masks["nucleus"]
    actin = masks["actin"]
    green = masks["green"]
    cyto = masks["cytoplasm"]
    cyto_no_actin = masks["cyto_no_actin"]
    cell = masks["cell"]
    background = masks["background"]
    green_actin_overlap = masks["green_actin_overlap"]
    green_nucleus_overlap = masks["green_nucleus_overlap"]
    actin_nucleus_overlap = masks["actin_nucleus_overlap"]
    tri_overlay_mask = masks["tri_overlay_mask"]
    rgb_disp = masks["rgb_disp"]

    R = img[..., 0].astype(np.float32)
    G = img[..., 1].astype(np.float32)
    B = img[..., 2].astype(np.float32)

    valid_corr_mask = cell & ((R + G + B) > 0)
    idx = valid_corr_mask.flatten()

    g_nuc = G[blue]
    g_act = G[actin]
    g_cyto = G[cyto]
    g_cyto_no_actin = G[cyto_no_actin]

    manders_nucleus = manders(G, blue)
    manders_actin = manders(G, actin)
    manders_cytoplasm = manders(G, cyto)
    manders_actin_free = manders(G, cyto_no_actin)
    manders_green_actin_overlap = manders(G, green_actin_overlap)
    manders_green_nucleus_overlap = manders(G, green_nucleus_overlap)
    manders_actin_nucleus_green = manders(G, actin_nucleus_overlap)

    ga_coloc_map = overlap_score(masks["G_s"], masks["R_s"] + masks["B_s"]) * cell.astype(np.float32)
    ga_only_coloc_map = overlap_score(masks["G_s"], masks["R_s"]) * cell.astype(np.float32)
    tri_coloc_map = overlap_score(masks["G_s"], masks["R_s"]) * overlap_score(masks["G_s"], masks["B_s"]) * cell.astype(np.float32)

    dist_to_actin = distance_transform_edt(~actin)
    green_pixels = green
    green_dist_vals = dist_to_actin[green_pixels]
    puncta_mask = masks["green_puncta"]
    filament_mask = masks["green_filament"]
    puncta_dist_vals = dist_to_actin[puncta_mask]
    filament_dist_vals = dist_to_actin[filament_mask]

    puncta_lab = label(puncta_mask)
    puncta_props = regionprops(puncta_lab)
    puncta_areas = [p.area for p in puncta_props]

    # Save masks
    save_mask(os.path.join(outdir, "mask_nucleus.png"), blue, "Nucleus mask", cmap="Blues")
    save_mask(os.path.join(outdir, "mask_nucleus_exclusion.png"), masks["nucleus_exclusion"], "Nucleus exclusion", cmap="Blues")
    save_heat_overlay(os.path.join(outdir, "red_sato_detection_response.png"), rgb_disp, masks["red_response"], "Red Sato response")
    save_mask(os.path.join(outdir, "mask_interacting_red.png"), masks["interacting_red"], "Interacting red area", cmap="YlOrBr")
    save_mask(os.path.join(outdir, "mask_actin.png"), actin, "Actin mask", cmap="Purples")
    save_mask(os.path.join(outdir, "mask_green.png"), green, "Green mask", cmap="Greens")
    save_mask(os.path.join(outdir, "mask_green_puncta.png"), puncta_mask, "Green puncta mask", cmap="Greens")
    save_mask(os.path.join(outdir, "mask_green_filament.png"), filament_mask, "Green filament mask", cmap="Greens")
    save_mask(os.path.join(outdir, "mask_cytoplasm.png"), cyto, "Cytoplasm mask", cmap="Greens")
    save_mask(os.path.join(outdir, "mask_actin_free_cytoplasm.png"), cyto_no_actin, "Actin-free cytoplasm", cmap="Oranges")
    save_mask(os.path.join(outdir, "mask_background.png"), background, "Background mask", cmap="gray")
    save_mask(os.path.join(outdir, "mask_green_actin_overlap.png"), green_actin_overlap, "Green-Actin overlap", cmap="magma")
    save_mask(os.path.join(outdir, "mask_green_nucleus_overlap.png"), green_nucleus_overlap, "Green-Nucleus overlap", cmap="cool")
    save_mask(os.path.join(outdir, "mask_actin_nucleus_overlap.png"), actin_nucleus_overlap, "Actin-Nucleus overlap", cmap="viridis")
    save_mask(os.path.join(outdir, "mask_triple_overlay_diagnostic.png"), tri_overlay_mask, "Triple overlap diagnostic", cmap="Reds")

    save_overlay(os.path.join(outdir, "overlay_masks.png"), rgb_disp, blue, actin, green)
    save_heat_overlay(os.path.join(outdir, "green_attention_map.png"), rgb_disp, G, "Green attention map")
    save_heat_overlay(os.path.join(outdir, "green_attention_map_cell_only.png"), rgb_disp, G * cell.astype(np.float32), "Green attention map (cell area only)")
    save_heat_overlay(os.path.join(outdir, "green_background_corrected_map.png"), rgb_disp, masks["G_corr"], "Green background-corrected map")
    save_heat_overlay(os.path.join(outdir, "green_frangi_map.png"), rgb_disp, masks["G_frangi"], "Green filament-enhanced map", cmap="inferno")
    save_heat_overlay(os.path.join(outdir, "green_actin_colocalization_map.png"), rgb_disp, ga_only_coloc_map, "Green-Actin colocalization map", cmap="inferno", alpha=0.65)
    save_heat_overlay(os.path.join(outdir, "green_actin_nucleus_colocalization_map.png"), rgb_disp, tri_coloc_map, "Green-Actin-Nucleus colocalization map", cmap="inferno", alpha=0.65)

    plot_distribution(g_nuc, g_act, g_cyto, g_cyto_no_actin, os.path.join(outdir, "green_distribution_violin.png"))
    plot_hexbin(B.flatten()[idx], G.flatten()[idx], "Blue intensity", "Green intensity", "Green vs Blue", os.path.join(outdir, "green_vs_blue_hexbin.png"))
    plot_hexbin(R.flatten()[idx], G.flatten()[idx], "Actin / purple-channel intensity", "Green intensity", "Green vs Actin", os.path.join(outdir, "green_vs_actin_hexbin.png"))

    dist_from_nucleus = distance_transform_edt(~blue) if np.any(blue) else np.zeros_like(blue, dtype=np.float32)
    dist_from_actin = distance_transform_edt(~actin)
    dist_from_actin_nucleus = distance_transform_edt(~(actin.astype(bool) | blue.astype(bool)))
    plot_radial_profile(dist_from_nucleus[cell], G[cell], os.path.join(outdir, "green_radial_profile.png"), "Radial green profile", "Mean green intensity")
    plot_radial_profile(dist_from_actin[cell], ga_only_coloc_map[cell], os.path.join(outdir, "green_actin_colocalization_radial_profile.png"), "Radial green-actin colocalization profile", "Mean green-actin colocalization")
    plot_radial_profile(dist_from_actin_nucleus[cell], tri_coloc_map[cell], os.path.join(outdir, "green_actin_nucleus_colocalization_radial_profile.png"), "Radial green-actin-nucleus colocalization profile", "Mean triple colocalization")

    row = {
        "file_index": file_index,
        "group": group,
        "image": name,
        "path": img_path,
        "file_type": Path(img_path).suffix.lower(),
        "height": img.shape[0],
        "width": img.shape[1],
        "green_nucleus_mean": safe_mean(g_nuc),
        "green_actin_mean": safe_mean(g_act),
        "green_cytoplasm_mean": safe_mean(g_cyto),
        "green_cytoplasm_no_actin_mean": safe_mean(g_cyto_no_actin),
        "green_triple_overlap_mean_diagnostic": safe_mean(G[tri_overlay_mask]),
        "nucleus_count": masks["nucleus_count"],
        "red_filament_count": masks["red_filament_count"],
        "green_intensity_region_count": masks["green_region_count"],
        "proximity_radius_pixels": INTERACTION_PROXIMITY_RADIUS,
        "total_red_filament_area_pixels": int(np.count_nonzero(actin)),
        "interacting_red_filament_area_pixels": int(np.count_nonzero(masks["interacting_red"])),
        "fraction_red_filament_area_interacting": masks["fraction_red_filament_area_interacting"],
        "nucleus_pixels": int(np.sum(blue)),
        "actin_pixels": int(np.sum(actin)),
        "green_pixels": int(np.sum(green)),
        "green_puncta_pixels": int(np.sum(puncta_mask)),
        "green_filament_pixels": int(np.sum(filament_mask)),
        "cytoplasm_pixels": int(np.sum(cyto)),
        "actin_free_cytoplasm_pixels": int(np.sum(cyto_no_actin)),
        "background_pixels": int(np.sum(background)),
        "green_actin_overlap_pixels": int(np.sum(green_actin_overlap)),
        "green_nucleus_overlap_pixels": int(np.sum(green_nucleus_overlap)),
        "actin_nucleus_overlap_pixels": int(np.sum(actin_nucleus_overlap)),
        "triple_overlay_pixels_diagnostic": int(np.sum(tri_overlay_mask)),
        "pearson_green_blue": safe_pearson(B.flatten()[idx], G.flatten()[idx]),
        "pearson_green_actin": safe_pearson(R.flatten()[idx], G.flatten()[idx]),
        "manders_green_nucleus": manders_nucleus,
        "manders_green_actin": manders_actin,
        "manders_green_cytoplasm": manders_cytoplasm,
        "manders_green_actin_free": manders_actin_free,
        "manders_green_actin_overlap": manders_green_actin_overlap,
        "manders_green_nucleus_overlap": manders_green_nucleus_overlap,
        "manders_actin_nucleus_green": manders_actin_nucleus_green,
        "mean_green_actin_coloc_score": safe_mean(ga_only_coloc_map[cell]),
        "mean_green_actin_nucleus_coloc_score": safe_mean(tri_coloc_map[cell]),
        "mean_green_to_actin_distance": safe_mean(green_dist_vals),
        "std_green_to_actin_distance": safe_std(green_dist_vals),
        "mean_puncta_to_actin_distance": safe_mean(puncta_dist_vals),
        "mean_filament_to_actin_distance": safe_mean(filament_dist_vals),
        "puncta_count": len(puncta_props),
        "mean_puncta_area": safe_mean(puncta_areas),
    }
    for r in PROXIMITY_RADII:
        row[f"fraction_green_within_{r}px_actin"] = float(np.mean(green_dist_vals <= r)) if green_dist_vals.size > 0 else np.nan
        row[f"fraction_puncta_within_{r}px_actin"] = float(np.mean(puncta_dist_vals <= r)) if puncta_dist_vals.size > 0 else np.nan
        row[f"fraction_filament_within_{r}px_actin"] = float(np.mean(filament_dist_vals <= r)) if filament_dist_vals.size > 0 else np.nan

    pd.DataFrame([row]).to_csv(os.path.join(outdir, "image_summary.csv"), index=False)
    region_df = pd.DataFrame([
        build_region_long_row(group, name, "Nucleus", g_nuc),
        build_region_long_row(group, name, "Actin", g_act),
        build_region_long_row(group, name, "Cytoplasm", g_cyto),
        build_region_long_row(group, name, "ActinFree", g_cyto_no_actin),
    ])
    region_df.to_csv(os.path.join(outdir, "image_region_summary.csv"), index=False)

    with open(os.path.join(outdir, "manders_values.txt"), "w") as f:
        f.write(f"manders_green_nucleus,{manders_nucleus}\n")
        f.write(f"manders_green_actin,{manders_actin}\n")
        f.write(f"manders_green_cytoplasm,{manders_cytoplasm}\n")
        f.write(f"manders_green_actin_free,{manders_actin_free}\n")
        f.write(f"manders_green_actin_overlap,{manders_green_actin_overlap}\n")
        f.write(f"manders_green_nucleus_overlap,{manders_green_nucleus_overlap}\n")
        f.write(f"manders_actin_nucleus_green,{manders_actin_nucleus_green}\n")

    pixel_df = sample_pixels_for_anova(name, group, G, blue, actin, cell)
    return row, region_df, pixel_df

# =========================
# Main
# =========================
def main():
    ensure_dir(OUT_DIR)
    all_files = []
    for g in GROUP_DIRS:
        folder = os.path.join(ROOT_DIR, g)
        all_files.extend(list_images(folder))
    all_files = sorted(p for p in all_files if not Path(p).resolve().is_relative_to(Path(OUT_DIR).resolve()))

    log_path = os.path.join(OUT_DIR, "run_log.txt")
    with open(log_path, "w") as log:
        log.write(f"ROOT_DIR: {ROOT_DIR}\n")
        log.write(f"OUT_DIR: {OUT_DIR}\n")
        log.write(f"Found {len(all_files)} image(s)\n\n")

        if len(all_files) == 0:
            print(f"No tif/tiff/png/jpg/jpeg files found under {ROOT_DIR}")
            return

        summary_rows = []
        region_dfs = []
        pixel_anova_parts = []

        for i, img_path in enumerate(all_files, 1):
            group = infer_group(img_path)
            print(f"[{i}/{len(all_files)}] {group} :: {Path(img_path).name}")
            log.write(f"[{i}/{len(all_files)}] {img_path}\n")
            try:
                row, region_df, pixel_df = analyze_image(img_path, i, OUT_DIR)
                summary_rows.append(row)
                region_dfs.append(region_df)
                pixel_anova_parts.append(pixel_df)
                log.write("  OK\n\n")
            except Exception as e:
                err = traceback.format_exc()
                print(f"  ERROR: {e}")
                log.write(f"  ERROR: {e}\n")
                log.write(err + "\n")

    if len(summary_rows) == 0:
        print("No images processed successfully.")
        return

    summary_df = pd.DataFrame(summary_rows)
    region_long_df = pd.concat(region_dfs, ignore_index=True) if region_dfs else pd.DataFrame()
    pixel_anova_df = pd.concat(pixel_anova_parts, ignore_index=True) if pixel_anova_parts else pd.DataFrame()

    summary_df.to_csv(os.path.join(OUT_DIR, "summary_all_images.csv"), index=False)
    region_long_df.to_csv(os.path.join(OUT_DIR, "region_long_summary.csv"), index=False)
    pixel_anova_df.to_csv(os.path.join(OUT_DIR, "pixel_sample_for_three_way_anova.csv"), index=False)

    build_group_summary(summary_df, os.path.join(OUT_DIR, "group_summary.csv"))
    stats_df = compare_groups(summary_df, os.path.join(OUT_DIR, "group_stats"))
    make_group_plots(summary_df, stats_df, os.path.join(OUT_DIR, "group_stats"))

    plot_summary_bar(summary_df, os.path.join(OUT_DIR, "SUMMARY_mean_green_by_region.png"))
    pixel_aov = run_three_way_anova(pixel_anova_df, OUT_DIR)
    region_aov, tukey_df = run_region_anova(region_long_df, OUT_DIR)
    tukey_strings = format_tukey_strings(tukey_df)
    plot_manders_boxplot(summary_df, os.path.join(OUT_DIR, "MANDERS_overlap_boxplot.png"), region_pvals=tukey_strings)

    with open(os.path.join(OUT_DIR, "STATS_SUMMARY.txt"), "w") as f:
        f.write("Three-way ANOVA on sampled cell pixels\n")
        f.write("Model: green_intensity ~ C(group) + C(image) + C(in_nucleus) * C(in_actin)\n\n")
        if pixel_aov is not None:
            f.write(pixel_aov.to_string(index=False))
        else:
            f.write("Three-way ANOVA failed or insufficient data.\n")
        f.write("\n\nRegion ANOVA on image-level means\n")
        f.write("Model: mean_green ~ C(group) + C(image) + C(region)\n\n")
        if region_aov is not None:
            f.write(region_aov.to_string(index=False))
        else:
            f.write("Region ANOVA failed or insufficient data.\n")
        f.write("\n\nTukey pairwise comparisons\n\n")
        if tukey_df is not None:
            f.write(tukey_df.to_string(index=False))
        else:
            f.write("Tukey test unavailable.\n")

    print("Done.")
    print(f"Results written to: {OUT_DIR}")
    print(f"Log written to: {os.path.join(OUT_DIR, 'run_log.txt')}")


if __name__ == "__main__":
    main()

