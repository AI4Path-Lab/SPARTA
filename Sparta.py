#!/usr/bin/env python3
import os
import glob
import math
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.ndimage import gaussian_filter, binary_fill_holes, distance_transform_edt
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import mannwhitneyu, ttest_ind, wasserstein_distance
from statsmodels.stats.multitest import multipletests

from skimage import io
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries, watershed
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, remove_small_holes
from cellpose import models
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT_DIR = "/fs/scratch/PAS2942/Alejandro/Shanaka/Cancer" # update this depending on where your images are.
OUT_DIR = "/fs/scratch/PAS2942/Alejandro/Shanaka/spatial_interaction_analysis"    #Change this if you want to save outputs somewhere else. 
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF")

GROUP_MAP = {"D": "Cancer", "C": "Normal"}
TYPE_COLORS = {"B_cell": "#16a34a", "Macrophage": "#dc2626", "Other": "#2563eb", "DoublePositive": "#a855f7"}
GROUP_COLORS = {"Cancer": "#dc2626", "Normal": "#2563eb"}

MACROPHAGE_SPATIAL_TYPES = ("Macrophage", "DoublePositive")
THRESH_VIO = 95
RNG_SEED = 7
CELLPOSE_MODEL_TYPE = "nuclei"
CELLPOSE_DIAMETER = None
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_CELLPROB_THRESHOLD = 0.0
NUCLEUS_MIN_AREA = 50
NUCLEUS_MAX_AREA = 5000
MARKER_RING_INNER = 3
MARKER_RING_OUTER = 5
MARKER_CALL_Q = 0.80
PROX_RADII = (10, 20, 30)
NN_BINS = 40
K_RADII = np.arange(5, 121, 5, dtype=float)
G_RADII = np.arange(5, 121, 5, dtype=float)
G_BANDWIDTH = 5.0
N_ENVELOPES = 39
MIN_TOTAL_CELLS = 30
MIN_B_CELLS = 5
MIN_M_CELLS = 5
VALIDATION_N_PATCHES_PER_IMAGE = 8
VALIDATION_PATCH_SIZE = 256
VALIDATION_SEED = 7
SEGMENTATION_MODEL = os.environ.get("SEGMENTATION_MODEL", "cellpose").strip().lower()  #### CHANGE BASELINES HERE, new folder made so there is no overlap between baselines.

BASE_OUT_DIR = OUT_DIR
OUT_DIR = os.path.join(BASE_OUT_DIR, SEGMENTATION_MODEL)

if SEGMENTATION_MODEL not in {"cellpose", "watershed", "stardist"}:
    raise ValueError(
        f"Unsupported SEGMENTATION_MODEL={SEGMENTATION_MODEL}. "
        "Use one of: cellpose, watershed, stardist."
    )

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(root_dir):
    files = []
    for ext in EXTS:
        files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    files = sorted(set(files))
    out = []
    for p in files:
        stem = Path(p).stem
        if stem and stem[0].upper() in GROUP_MAP:
            out.append(p)
    return out


def infer_group(path):
    return GROUP_MAP.get(Path(path).stem[0].upper(), "Unknown")


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def robust_rgb(img):
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.moveaxis(img, 0, -1)
    img = img.astype(np.float32)
    if img.max() > 0:
        img = img / img.max()
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        out[..., c] = normalize01(img[..., c])
    return out


def load_image(path):
    img = io.imread(path)
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[-1] >= 3:
            img = img[..., :3]
        elif img.shape[0] >= 3:
            img = np.moveaxis(img[:3, ...], 0, -1)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    img = img.astype(np.float32)
    if img.max() > 0:
        img = img / img.max()
    return img


def segment_nuclei_cellpose(img):
    try:
        from cellpose import models
    except Exception as e:
        raise ImportError(
            "Cellpose is required for this script. Install it in your environment, for example: pip install cellpose"
        ) from e

    blue = normalize01(gaussian_filter(img[..., 2], sigma=1.0))
    rgb = robust_rgb(img)
    model = models.CellposeModel(gpu=False)
    masks, flows, styles = model.eval(
        rgb,
        diameter=CELLPOSE_DIAMETER,
        flow_threshold=CELLPOSE_FLOW_THRESHOLD,
        cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
    )
    masks = np.asarray(masks, dtype=np.int32)
    props = pd.DataFrame(regionprops_table(
        masks,
        intensity_image=blue,
        properties=["label", "area", "centroid", "eccentricity", "mean_intensity"]
    ))
    if props.empty:
        return np.zeros_like(masks, dtype=np.int32), props
    props = props[(props["area"] >= NUCLEUS_MIN_AREA) & (props["area"] <= NUCLEUS_MAX_AREA)].copy()
    
    keep = set(props["label"].astype(int).tolist())
    masks = np.where(np.isin(masks, list(keep)), masks, 0).astype(np.int32)
    props = pd.DataFrame(regionprops_table(
        masks,
        intensity_image=blue,
        properties=["label", "area", "centroid", "eccentricity", "mean_intensity"]
    ))
    return masks, props

def props_from_nuclei_label(masks, blue):
    masks = np.asarray(masks, dtype=np.int32)

    props = pd.DataFrame(regionprops_table(
        masks,
        intensity_image=blue,
        properties=["label", "area", "centroid", "eccentricity", "mean_intensity"]
    ))

    if props.empty:
        return np.zeros_like(masks, dtype=np.int32), props

    props = props[
        (props["area"] >= NUCLEUS_MIN_AREA) &
        (props["area"] <= NUCLEUS_MAX_AREA)
    ].copy()

    keep = set(props["label"].astype(int).tolist())
    masks = np.where(np.isin(masks, list(keep)), masks, 0).astype(np.int32)

    props = pd.DataFrame(regionprops_table(
        masks,
        intensity_image=blue,
        properties=["label", "area", "centroid", "eccentricity", "mean_intensity"]
    ))

    return masks, props


def segment_nuclei_watershed(img):
    blue = normalize01(gaussian_filter(img[..., 2], sigma=1.0))

    try:
        thr = threshold_otsu(blue)
    except Exception:
        return np.zeros(img.shape[:2], dtype=np.int32), pd.DataFrame()

    mask = blue > thr
    mask = remove_small_objects(mask, min_size=NUCLEUS_MIN_AREA)
    mask = remove_small_holes(mask, area_threshold=NUCLEUS_MIN_AREA)
    mask = binary_fill_holes(mask).astype(bool)

    if not np.any(mask):
        return np.zeros(img.shape[:2], dtype=np.int32), pd.DataFrame()

    dist = distance_transform_edt(mask)

    coords = peak_local_max(
        dist,
        min_distance=5,
        labels=mask,
        exclude_border=False
    )

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i

    if markers.max() == 0:
        markers = mask.astype(np.int32)

    masks = watershed(-dist, markers, mask=mask)
    return props_from_nuclei_label(masks, blue)


_STARDIST_MODEL = None

def segment_nuclei_stardist(img):
    global _STARDIST_MODEL

    try:
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize as stardist_normalize
    except Exception as e:
        raise ImportError(
            "StarDist is required for SEGMENTATION_MODEL=stardist. "
            "Install with: pip install stardist csbdeep tensorflow"
        ) from e

    red = normalize01(gaussian_filter(img[..., 0], sigma=1.0))
    green = normalize01(gaussian_filter(img[..., 1], sigma=1.0))
    blue = normalize01(gaussian_filter(img[..., 2], sigma=1.0))

    # Match Cellpose better: use all visible signal, not blue only.
    composite = np.maximum.reduce([red, green, blue])
    composite = normalize01(composite)

    stardist_input = stardist_normalize(composite, 1, 99.8)

    if _STARDIST_MODEL is None:
        _STARDIST_MODEL = StarDist2D.from_pretrained("2D_versatile_fluo")

    masks, _ = _STARDIST_MODEL.predict_instances(stardist_input)

    return props_from_nuclei_label(masks, composite)


def segment_nuclei(img):
    if SEGMENTATION_MODEL == "cellpose":
        return segment_nuclei_cellpose(img)

    if SEGMENTATION_MODEL == "watershed":
        return segment_nuclei_watershed(img)

    if SEGMENTATION_MODEL == "stardist":
        return segment_nuclei_stardist(img)

    raise ValueError(f"Unsupported SEGMENTATION_MODEL={SEGMENTATION_MODEL}")


def ring_mask(shape, center_r, center_c, inner, outer):
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    d2 = (rr - center_r) ** 2 + (cc - center_c) ** 2
    return (d2 >= inner * inner) & (d2 <= outer * outer)


def call_cell_types(img, nuclei_props, nuclei_label):
    green = normalize01(gaussian_filter(img[..., 1], sigma=1.0))
    red = normalize01(gaussian_filter(img[..., 0], sigma=1.0))
    if nuclei_props.empty:
        return pd.DataFrame(columns=[
            "nucleus_label", "y", "x", "area", "blue_mean", "green_ring_mean", "red_ring_mean",
            "green_ring_max", "red_ring_max", "green_positive", "red_positive", "cell_type"
        ])

    global_green_thr = np.quantile(green, MARKER_CALL_Q)
    global_red_thr = np.quantile(red, MARKER_CALL_Q)
    green_hi = np.quantile(green, 0.97)
    red_hi = np.quantile(red, 0.97)
    rows = []
    for _, row in nuclei_props.iterrows():
        lab = int(row["label"])
        y = float(row["centroid-0"])
        x = float(row["centroid-1"])
        rm = ring_mask(img.shape[:2], y, x, MARKER_RING_INNER, MARKER_RING_OUTER)
        local_nucleus = nuclei_label == lab
        local_ring = rm & (~local_nucleus)
        if np.sum(local_ring) == 0:
            continue
        gvals = green[local_ring]
        rvals = red[local_ring]
        gmean = float(np.mean(gvals))
        rmean = float(np.mean(rvals))
        gmax = float(np.quantile(gvals, 0.95))
        rmax = float(np.quantile(rvals, 0.95))
        gpos = (gmean >= global_green_thr) or (gmax >= green_hi)
        rpos = (rmean >= global_red_thr) or (rmax >= red_hi)
        if gpos and rpos and (gmean > 1.2*global_green_thr) and (rmean > 1.2*global_red_thr):
            ctype = "DoublePositive"
        elif gpos:
            ctype = "B_cell"
        elif rpos:
            ctype = "Macrophage"
        else:
            ctype = "Other"
        rows.append({
            "nucleus_label": lab,
            "y": y,
            "x": x,
            "area": float(row["area"]),
            "blue_mean": float(row["mean_intensity"]),
            "green_ring_mean": gmean,
            "red_ring_mean": rmean,
            "green_ring_max": gmax,
            "red_ring_max": rmax,
            "green_positive": int(gpos),
            "red_positive": int(rpos),
            "cell_type": ctype,
        })
    return pd.DataFrame(rows)


def get_points(cell_df, cell_type):
    if isinstance(cell_type, (list, tuple, set)):
        sub = cell_df[cell_df["cell_type"].isin(cell_type)]
    else:
        sub = cell_df[cell_df["cell_type"] == cell_type]
    if sub.empty:
        return np.empty((0, 2), dtype=float)
    return sub[["x", "y"]].to_numpy(dtype=float)


def nearest_neighbor_distances(src, tgt, allow_self=False):
    src = np.asarray(src, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    if len(src) == 0 or len(tgt) == 0:
        return np.array([], dtype=float)
    tree = cKDTree(tgt)
    if allow_self and src.shape == tgt.shape and np.allclose(src, tgt):
        d, _ = tree.query(src, k=2)
        return d[:, 1].astype(float)
    d, _ = tree.query(src, k=1)
    return np.asarray(d, dtype=float)


def pair_counts_cross(src, tgt):
    if len(src) == 0 or len(tgt) == 0:
        return np.array([], dtype=float)
    return distance_matrix(src, tgt)


def cross_ripley_k(src, tgt, area, radii):
    src = np.asarray(src, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    if len(src) == 0 or len(tgt) == 0 or area <= 0:
        return np.full(len(radii), np.nan, dtype=float)
    dm = pair_counts_cross(src, tgt)
    norm = area / (len(src) * len(tgt))
    return np.asarray([norm * np.sum(dm <= r) for r in radii], dtype=float)


def pair_correlation_cross(src, tgt, area, radii, dr):
    src = np.asarray(src, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    if len(src) == 0 or len(tgt) == 0 or area <= 0:
        return np.full(len(radii), np.nan, dtype=float)
    dm = pair_counts_cross(src, tgt)
    lam_t = len(tgt) / area
    out = []
    for r in radii:
        lo = max(0.0, r - dr / 2.0)
        hi = r + dr / 2.0
        shell = (dm > lo) & (dm <= hi)
        expected = len(src) * lam_t * math.pi * (hi * hi - lo * lo)
        out.append(np.sum(shell) / expected if expected > 0 else np.nan)
    return np.asarray(out, dtype=float)


def random_points_in_box(n, width, height, rng):
    return np.column_stack([rng.uniform(0, width, size=n), rng.uniform(0, height, size=n)])


def csr_envelopes(n_src, n_tgt, width, height, radii, n_env, which, rng):
    curves = []
    area = width * height
    for _ in range(n_env):
        src = random_points_in_box(n_src, width, height, rng)
        tgt = random_points_in_box(n_tgt, width, height, rng)
        if which == "K":
            curves.append(cross_ripley_k(src, tgt, area, radii))
        else:
            curves.append(pair_correlation_cross(src, tgt, area, radii, G_BANDWIDTH))
    curves = np.asarray(curves, dtype=float)
    return (
        np.nanmedian(curves, axis=0),
        np.nanpercentile(curves, 2.5, axis=0),
        np.nanpercentile(curves, 97.5, axis=0),
    )


def effect_size(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    if sp < 1e-12:
        return np.nan
    return (np.mean(a) - np.mean(b)) / sp


def bootstrap_ci_delta(a, b, n_boot=2000, seed=7):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        vals.append(np.mean(aa) - np.mean(bb))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def scalar_from_curve(y, x, mode):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if np.all(~np.isfinite(y)):
        return np.nan
    if mode == "auc":
        return float(np.trapz(np.nan_to_num(y, nan=0.0), x))
    if mode == "peak":
        return float(np.nanmax(y))
    if mode == "mean":
        return float(np.nanmean(y))
    return np.nan


def save_qc_figure(img, nuclei_label, cell_df, out_path, title):
    rgb = robust_rgb(img)
    overlay = rgb.copy()
    bounds = find_boundaries(nuclei_label > 0, mode="outer")
    overlay[bounds] = np.array([1.0, 1.0, 0.0])
    fig, ax = plt.subplots(figsize=(8, 8),sharey=True)
    ax.imshow(overlay)
    if not cell_df.empty:
        for ctype in ["B_cell", "Macrophage", "DoublePositive", "Other"]:
            sub = cell_df[cell_df["cell_type"] == ctype]
            if not sub.empty:
                ax.scatter(sub["x"], sub["y"], s=18, c=TYPE_COLORS[ctype], label=ctype, edgecolors="white", linewidths=0.3)
    ax.set_title(title)
    ax.axis("off")
    handles = [Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=8, markeredgecolor='white') for k, v in TYPE_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_representative_panel(reps, out_path):
    fig, axes = plt.subplots(1, len(reps), figsize=(7 * len(reps), 7), sharey = True)
    if len(reps) == 1:
        axes = [axes]
    for ax, rep in zip(axes, reps):
        img = load_image(rep["path"])
        ax.imshow(robust_rgb(img))
        ax.set_title(f"{rep['group']}\n{rep['image']}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_curve_with_envelope(curves_df, metric_name, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 6), sharey=True)
    for group in ["Normal", "Cancer"]:
        sub = curves_df[curves_df["group"] == group].copy()
        if sub.empty:
            continue
        x = np.sort(sub["radius"].unique())
        y_mean, y_lo, y_hi, env_lo, env_hi = [], [], [], [], []
        for r in x:
            sr = sub[sub["radius"] == r]
            y_mean.append(np.nanmean(sr[metric_name]))
            y_lo.append(np.nanpercentile(sr[metric_name], 25))
            y_hi.append(np.nanpercentile(sr[metric_name], 75))
            if f"{metric_name}_env_lo" in sr.columns:
                env_lo.append(np.nanmean(sr[f"{metric_name}_env_lo"]))
                env_hi.append(np.nanmean(sr[f"{metric_name}_env_hi"]))
        color = GROUP_COLORS[group]
        ax.plot(x, y_mean, color=color, linewidth=2.5, label=group)
        ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.18)
        if len(env_lo) == len(x):
            ax.plot(x, env_lo, color=color, linestyle="--", alpha=0.45)
            ax.plot(x, env_hi, color=color, linestyle="--", alpha=0.45)
    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend()
    ax.set_yticks(np.arange(0, 80000, 10000))
    ax.set_ylim(0,70000)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
def plot_curve_with_envelope2(curves_df, metric_name, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 6), sharey=True)
    for group in ["Normal", "Cancer"]:
        sub = curves_df[curves_df["group"] == group].copy()
        if sub.empty:
            continue
        x = np.sort(sub["radius"].unique())
        y_mean, y_lo, y_hi, env_lo, env_hi = [], [], [], [], []
        for r in x:
            sr = sub[sub["radius"] == r]
            y_mean.append(np.nanmean(sr[metric_name]))
            y_lo.append(np.nanpercentile(sr[metric_name], 25))
            y_hi.append(np.nanpercentile(sr[metric_name], 75))
            if f"{metric_name}_env_lo" in sr.columns:
                env_lo.append(np.nanmean(sr[f"{metric_name}_env_lo"]))
                env_hi.append(np.nanmean(sr[f"{metric_name}_env_hi"]))
        color = GROUP_COLORS[group]
        ax.plot(x, y_mean, color=color, linewidth=2.5, label=group)
        ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.18)
        if len(env_lo) == len(x):
            ax.plot(x, env_lo, color=color, linestyle="--", alpha=0.45)
            ax.plot(x, env_hi, color=color, linestyle="--", alpha=0.45)
    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend()
    ax.set_yticks(np.arange(0, 2.2, 0.2))
    ax.set_ylim(0,2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests  
import seaborn as sns
     
def plot_nn_violin(nn_image_long, nn_cell_long, out_path):

    pairs = ["B_to_M", "B_to_B", "M_to_M"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey="row")

    def pstat(p_value):
        if p_value < 0.0001:
            return "****"
        elif p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"

    def test(ax, x1, x2, y, h, text):
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            color="black",
            linewidth=1.5,
        )
        ax.text(
            (x1 + x2) / 2,
            y + h,
            text,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Image-level statistical tests
    raw_pvals = []

    for pair in pairs:
        sub_img = nn_image_long[nn_image_long["pair"] == pair]

        normal = (
            sub_img.loc[sub_img["group"] == "Normal", "distance"]
            .dropna()
            .to_numpy(dtype=float)
        )

        cancer = (
            sub_img.loc[sub_img["group"] == "Cancer", "distance"]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(normal) > 0 and len(cancer) > 0:
            _, p = mannwhitneyu(
                normal,
                cancer,
                alternative="two-sided",
            )
        else:
            p = np.nan

        raw_pvals.append(p)

    raw_pvals = np.asarray(raw_pvals, dtype=float)

    valid = ~np.isnan(raw_pvals)
    adjusted = np.full(len(raw_pvals), np.nan)

    if np.any(valid):
        adjusted[valid] = multipletests(
            raw_pvals[valid],
            method="holm",
        )[1]
# Maximum value across ALL cell-level plots
    cell_max = np.percentile(nn_cell_long["distance"].dropna(), THRESH_VIO)

    # Maximum value across ALL image-level plots
    img_max = nn_image_long["distance"].dropna().max()

    # Round up to next 10, then add another 10 for padding
    cell_ylim = np.ceil(cell_max / 10) * 10 + 10
    img_ylim = np.ceil(img_max / 10) * 10 + 10
    # Plotting
    for j, pair in enumerate(pairs):
        sub_cell = nn_cell_long[nn_cell_long["pair"] == pair].copy()
        sub_img = nn_image_long[nn_image_long["pair"] == pair].copy()

        p_adj = adjusted[j]
        
        vals = sub_cell["distance"].dropna().to_numpy(dtype=float)

        if len(vals) > 0:
            threshold = np.percentile(vals, THRESH_VIO)

            sub_cell = sub_cell[
                sub_cell["distance"] <= threshold
            ]

        if len(sub_cell) > 30000:
            sub_cell = sub_cell.sample(
                30000,
                random_state=RNG_SEED,
            )

        # Cell-level violin plot
        ax = axes[0, j]

        if sub_cell.empty:
            ax.axis("off")

        else:
            sns.violinplot(
                data=sub_cell,
                x="group",
                y="distance",
                order=["Normal", "Cancer"],
                palette=[
                    GROUP_COLORS["Normal"],
                    GROUP_COLORS["Cancer"],
                ],
                cut=0,
                inner="quartile",
                ax=ax,
            )

            ax.set_title(
                pair.replace("_", " ") +
                "\nCell-level NN distances"
            )
            ax.set_xlabel("")
            ax.set_ylabel("NN distance (pixels)")

            if not np.isnan(p_adj):
                test(
                    ax,
                    x1=0,
                    x2=1,
                    y=cell_ylim - 8,
                    h=3,
                    text=pstat(p_adj),
                )

        # Image-level plot
        ax = axes[1, j]

        if sub_img.empty:
            ax.axis("off")

        else:
            sns.stripplot(
                data=sub_img,
                x="group",
                y="distance",
                order=["Normal", "Cancer"],
                color="black",
                size=6,
                jitter=0.12,
                ax=ax,
            )

            for xpos, group in enumerate(["Normal", "Cancer"]):
                vals = (
                    sub_img.loc[
                        sub_img["group"] == group,
                        "distance",
                    ]
                    .dropna()
                    .to_numpy(dtype=float)
                )

                if len(vals) == 0:
                    continue

                mean = float(np.mean(vals))

                if len(vals) >= 2:
                    rng = np.random.default_rng(RNG_SEED)

                    boots = [
                        np.mean(
                            rng.choice(
                                vals,
                                size=len(vals),
                                replace=True,
                            )
                        )
                        for _ in range(2000)
                    ]

                    lo, hi = np.percentile(
                        boots,
                        [2.5, 97.5],
                    )

                else:
                    lo, hi = mean, mean

                ax.errorbar(
                    xpos,
                    mean,
                    yerr=[
                        [mean - lo],
                        [hi - mean],
                    ],
                    fmt="_",
                    color="gray",
                    markersize=22,
                    capsize=5,
                    linewidth=2,
                )

            n_norm = int(
                sub_img[
                    sub_img["group"] == "Normal"
                ]["image"].nunique()
            )

            n_canc = int(
                sub_img[
                    sub_img["group"] == "Cancer"
                ]["image"].nunique()
            )

            ax.set_title(
                f"Image-level means\n"
                f"Normal n={n_norm}, Cancer n={n_canc}"
            )
            ax.set_xlabel("")
            ax.set_ylabel("Per-image mean NN distance")

            # Same image-level test shown here
            if not np.isnan(p_adj):
                test(
                    ax,
                    x1=0,
                    x2=1,
                    y=img_ylim - 8,
                    h=3,
                    text=pstat(p_adj),
                )

    # Top row
    for a in axes[0, :]:
        a.set_ylim(0, cell_ylim)
        a.set_yticks(np.arange(0, cell_ylim + 1, 10))

    # Bottom row
    for a in axes[1, :]:
        a.set_ylim(0, img_ylim)
        a.set_yticks(np.arange(0, img_ylim + 1, 10))

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_effect_forest(stats_df, out_path):
    sub = stats_df.sort_values("fdr").head(18).copy()
    if sub.empty:
        return
    sub = sub.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.38 * len(sub) + 1)),sharey = True)
    y = np.arange(len(sub))
    ax.errorbar(sub["delta_mean"], y, xerr=[sub["delta_mean"] - sub["ci_lo"], sub["ci_hi"] - sub["delta_mean"]], fmt='o', color='black', ecolor='gray', capsize=3)
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["metric"])
    ax.set_xlabel("Cancer minus Normal")
    ax.set_title("Top image-level differences")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(metrics_df, metric_cols, out_path):
    import seaborn as sns
    use = metrics_df[["image", "group"] + metric_cols].copy()
    if use.empty:
        return
    use = use.set_index(["group", "image"])
    z = use.apply(lambda c: (c - c.mean()) / (c.std(ddof=0) + 1e-8), axis=0)
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(metric_cols) + 3), max(6, 0.3 * len(z) + 2)), sharey = True)
    sns.heatmap(z, cmap="vlag", center=0, ax=ax)
    ax.set_title("Spatial metric heatmap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_group_tests(df, id_cols=("image", "group")):
    out = []
    numeric_cols = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        a = df.loc[df["group"] == "Cancer", col].dropna().to_numpy(dtype=float)
        b = df.loc[df["group"] == "Normal", col].dropna().to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2:
            continue
        tw = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        mw = mannwhitneyu(a, b, alternative="two-sided")
        ci_lo, ci_hi = bootstrap_ci_delta(a, b, seed=RNG_SEED)
        out.append({
            "metric": col,
            "cancer_n": len(a),
            "normal_n": len(b),
            "cancer_mean": float(np.mean(a)),
            "normal_mean": float(np.mean(b)),
            "delta_mean": float(np.mean(a) - np.mean(b)),
            "welch_p": float(tw.pvalue),
            "mw_p": float(mw.pvalue),
            "effect_size_d": effect_size(a, b),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })
    out = pd.DataFrame(out)
    if not out.empty:
        out["fdr"] = multipletests(out["welch_p"], method="fdr_bh")[1]
        out = out.sort_values(["fdr", "welch_p", "metric"]).reset_index(drop=True)
    return out


def compute_template_emd(per_image_dists, bins):
    rows = []
    hist_map = {}
    for key, arr in per_image_dists.items():
        h, _ = np.histogram(arr, bins=bins, density=True)
        h = h.astype(float)
        if h.sum() > 0:
            h = h / h.sum()
        hist_map[key] = h
    mids = 0.5 * (bins[:-1] + bins[1:])
    for (img, group), h in hist_map.items():
        cancer_h = [v for (i, g), v in hist_map.items() if g == "Cancer" and i != img]
        normal_h = [v for (i, g), v in hist_map.items() if g == "Normal" and i != img]
        cancer_template = np.mean(cancer_h, axis=0) if len(cancer_h) > 0 else np.full_like(h, np.nan)
        normal_template = np.mean(normal_h, axis=0) if len(normal_h) > 0 else np.full_like(h, np.nan)
        rows.append({
            "image": img,
            "group": group,
            "emd_to_cancer_template": float(wasserstein_distance(mids, mids, u_weights=h, v_weights=cancer_template)) if np.isfinite(cancer_template).all() else np.nan,
            "emd_to_normal_template": float(wasserstein_distance(mids, mids, u_weights=h, v_weights=normal_template)) if np.isfinite(normal_template).all() else np.nan,
        })
    return pd.DataFrame(rows)

def save_validation_patches(img, cell_df, image_dir, image_name, patch_size=256, n_patches=8):
    if cell_df.empty:
        return pd.DataFrame()

    h, w = img.shape[:2]
    half = patch_size // 2
    rows = []

    candidates = cell_df.copy()
    if len(candidates) > n_patches:
        candidates = candidates.sample(n=n_patches, random_state=VALIDATION_SEED)

    rgb = robust_rgb(img)

    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        yc = int(round(row["y"]))
        xc = int(round(row["x"]))

        y0 = max(0, yc - half)
        y1 = min(h, yc + half)
        x0 = max(0, xc - half)
        x1 = min(w, xc + half)

        patch = rgb[y0:y1, x0:x1].copy()

        fig, ax = plt.subplots(figsize=(5, 5), sharey = True)
        ax.imshow(patch)
        ax.scatter([xc - x0], [yc - y0], s=30, c="yellow", edgecolors="black", linewidths=0.5)
        ax.set_title(f"{image_name} patch {i} | pred={row['cell_type']}")
        ax.axis("off")

        patch_name = f"{image_name}_patch_{i:02d}.png"
        patch_path = os.path.join(image_dir, patch_name)
        fig.tight_layout()
        fig.savefig(patch_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

        rows.append({
            "image": image_name,
            "patch_id": i,
            "patch_file": patch_name,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "predicted_label": row["cell_type"],
            "manual_label": "",
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(image_dir, f"{image_name}_validation_manifest.csv"), index=False)
    return out

def summarize_detection_vs_manual(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "validation_patches", "*_manual_counts.csv")))
    if len(files) == 0:
        return

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    needed = [
        "image", "manual_nuclei", "pred_nuclei",
        "manual_b_cells", "pred_b_cells",
        "manual_macrophages", "pred_macrophages"
    ]
    if not all(c in df.columns for c in needed):
        return

    rows = []
    for mcol, pcol, name in [
        ("manual_nuclei", "pred_nuclei", "nuclei"),
        ("manual_b_cells", "pred_b_cells", "B_cell"),
        ("manual_macrophages", "pred_macrophages", "Macrophage"),
    ]:
        x = df[mcol].to_numpy(dtype=float)
        y = df[pcol].to_numpy(dtype=float)
        ae = np.abs(y - x)
        pe = 100.0 * ae / np.maximum(x, 1.0)

        rows.append({
            "target": name,
            "n_regions": len(df),
            "manual_mean": float(np.mean(x)),
            "pred_mean": float(np.mean(y)),
            "mean_abs_error": float(np.mean(ae)),
            "mean_percent_error": float(np.mean(pe)),
            "pearson_r": float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan,
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "tables", "validation_detection_summary.csv"), index=False)
    
def save_manual_count_templates(img, cell_df, image_dir, image_name, patch_size=256, n_patches=8):
    if cell_df.empty:
        return

    h, w = img.shape[:2]
    half = patch_size // 2
    candidates = cell_df.copy()
    if len(candidates) > n_patches:
        candidates = candidates.sample(n=n_patches, random_state=VALIDATION_SEED)

    rows = []
    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        yc = int(round(row["y"]))
        xc = int(round(row["x"]))

        y0 = max(0, yc - half)
        y1 = min(h, yc + half)
        x0 = max(0, xc - half)
        x1 = min(w, xc + half)

        sub = cell_df[
            (cell_df["x"] >= x0) & (cell_df["x"] < x1) &
            (cell_df["y"] >= y0) & (cell_df["y"] < y1)
        ]

        rows.append({
            "image": image_name,
            "patch_id": i,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "pred_nuclei": int(len(sub)),
            "pred_b_cells": int(np.sum(sub["cell_type"] == "B_cell")),
            "pred_macrophages": int(np.sum(sub["cell_type"] == "Macrophage")),
            "manual_nuclei": "",
            "manual_b_cells": "",
            "manual_macrophages": "",
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(image_dir, f"{image_name}_manual_counts.csv"),
        index=False
    )


def analyze_one_image(path, out_dir, rng):
    group = infer_group(path)
    image = Path(path).stem
    img = load_image(path)
    h, w = img.shape[:2]
    area = float(h * w)
    image_dir = os.path.join(out_dir, "per_image", image)
    ensure_dir(image_dir)

    nuclei_label, nuclei_props = segment_nuclei(img)
    cell_df = call_cell_types(img, nuclei_props, nuclei_label)
    print(image, len(cell_df), np.sum(cell_df["cell_type"] == "B_cell"), np.sum(cell_df["cell_type"] == "Macrophage"))
    cell_df.insert(0, "image", image)
    cell_df.insert(1, "group", group)
    cell_df.insert(2, "path", path)
    cell_df.insert(3, "segmentation_model", SEGMENTATION_MODEL)
    cell_df.to_csv(os.path.join(image_dir, "cell_calls.csv"), index=False)
    validation_dir = os.path.join(out_dir, "validation_patches")
    ensure_dir(validation_dir)
    save_validation_patches(
        img=img,
        cell_df=cell_df,
        image_dir=validation_dir,
        image_name=image,
        patch_size=VALIDATION_PATCH_SIZE,
        n_patches=VALIDATION_N_PATCHES_PER_IMAGE,
    )
    save_manual_count_templates(
        img=img,
        cell_df=cell_df,
        image_dir=validation_dir,
        image_name=image,
        patch_size=VALIDATION_PATCH_SIZE,
        n_patches=VALIDATION_N_PATCHES_PER_IMAGE,
    )
    save_qc_figure(img, nuclei_label, cell_df, os.path.join(image_dir, "qc_cell_calls.png"), f"{image} ({group})")

    pts_b = get_points(cell_df, "B_cell")
    pts_m = get_points(cell_df, MACROPHAGE_SPATIAL_TYPES)

    b_to_m = nearest_neighbor_distances(pts_b, pts_m, allow_self=False)
    b_to_b = nearest_neighbor_distances(pts_b, pts_b, allow_self=True)
    m_to_m = nearest_neighbor_distances(pts_m, pts_m, allow_self=True)

    k_curve = cross_ripley_k(pts_b, pts_m, area, K_RADII)
    g_curve = pair_correlation_cross(pts_b, pts_m, area, G_RADII, G_BANDWIDTH)
    k_env_med, k_env_lo, k_env_hi = csr_envelopes(len(pts_b), len(pts_m), w, h, K_RADII, N_ENVELOPES, "K", rng)
    g_env_med, g_env_lo, g_env_hi = csr_envelopes(len(pts_b), len(pts_m), w, h, G_RADII, N_ENVELOPES, "g", rng)

    curve_rows = []
    for r, kv, km, klo, khi in zip(K_RADII, k_curve, k_env_med, k_env_lo, k_env_hi):
        curve_rows.append({"image": image, "group": group, "metric": "ripley_k", "radius": r, "ripley_k": kv, "ripley_k_env_med": km, "ripley_k_env_lo": klo, "ripley_k_env_hi": khi})
    for r, gv, gm, glo, ghi in zip(G_RADII, g_curve, g_env_med, g_env_lo, g_env_hi):
        curve_rows.append({"image": image, "group": group, "metric": "pair_correlation", "radius": r, "pair_correlation": gv, "pair_correlation_env_med": gm, "pair_correlation_env_lo": glo, "pair_correlation_env_hi": ghi})
    curves_df = pd.DataFrame(curve_rows)
    curves_df.to_csv(os.path.join(image_dir, "curves.csv"), index=False)

    summary = {
        "image": image,
        "group": group,
        "path": path,
        "segmentation_model": SEGMENTATION_MODEL,
        "height": h,
        "width": w,
        "area_px": area,
        "n_nuclei": int(len(cell_df)),
        "n_b_cells": int(np.sum(cell_df["cell_type"] == "B_cell")) if not cell_df.empty else 0,
        "n_macrophages": int(np.sum(cell_df["cell_type"] == "Macrophage")) if not cell_df.empty else 0,
        "n_double_positive": int(np.sum(cell_df["cell_type"] == "DoublePositive")) if not cell_df.empty else 0,
        "n_macrophage_spatial": int(np.sum(cell_df["cell_type"].isin(MACROPHAGE_SPATIAL_TYPES))) if not cell_df.empty else 0,
        "density_all": float(len(cell_df) / area) if area > 0 else np.nan,
        "density_b": float(len(pts_b) / area) if area > 0 else np.nan,
        "density_m": float(len(pts_m) / area) if area > 0 else np.nan,
        "mean_b_to_m": float(np.mean(b_to_m)) if len(b_to_m) > 0 else np.nan,
        "median_b_to_m": float(np.median(b_to_m)) if len(b_to_m) > 0 else np.nan,
        "mean_b_to_b": float(np.mean(b_to_b)) if len(b_to_b) > 0 else np.nan,
        "mean_m_to_m": float(np.mean(m_to_m)) if len(m_to_m) > 0 else np.nan,
        "ripley_auc": scalar_from_curve(k_curve, K_RADII, "auc"),
        "ripley_peak": scalar_from_curve(k_curve, K_RADII, "peak"),
        "ripley_mean": scalar_from_curve(k_curve, K_RADII, "mean"),
        "paircorr_auc": scalar_from_curve(g_curve, G_RADII, "auc"),
        "paircorr_peak": scalar_from_curve(g_curve, G_RADII, "peak"),
        "paircorr_mean": scalar_from_curve(g_curve, G_RADII, "mean"),
        "ripley_excess_auc": scalar_from_curve(k_curve - k_env_med, K_RADII, "auc") if np.isfinite(k_env_med).any() else np.nan,
        "paircorr_excess_auc": scalar_from_curve(g_curve - g_env_med, G_RADII, "auc") if np.isfinite(g_env_med).any() else np.nan,
    }
    for r in PROX_RADII:
        summary[f"frac_b_to_m_le_{r}"] = float(np.mean(b_to_m <= r)) if len(b_to_m) > 0 else np.nan
        summary[f"frac_b_to_b_le_{r}"] = float(np.mean(b_to_b <= r)) if len(b_to_b) > 0 else np.nan
        summary[f"frac_m_to_m_le_{r}"] = float(np.mean(m_to_m <= r)) if len(m_to_m) > 0 else np.nan

    nn_rows = []
    if len(b_to_m) > 0:
        nn_rows.append({"image": image, "group": group, "pair": "B_to_M", "distance": float(np.mean(b_to_m)), "n_cells": len(b_to_m)})
    if len(b_to_b) > 0:
        nn_rows.append({"image": image, "group": group, "pair": "B_to_B", "distance": float(np.mean(b_to_b)), "n_cells": len(b_to_b)})
    if len(m_to_m) > 0:
        nn_rows.append({"image": image, "group": group, "pair": "M_to_M", "distance": float(np.mean(m_to_m)), "n_cells": len(m_to_m)})

    return summary, pd.DataFrame(nn_rows), curves_df, cell_df, {
        "B_to_M": b_to_m.astype(float),
        "B_to_B": b_to_b.astype(float),
        "M_to_M": m_to_m.astype(float),
    }


def choose_representatives(metrics_df):
    reps = []
    for group in ["Normal", "Cancer"]:
        sub = metrics_df[metrics_df["group"] == group].copy()
        if sub.empty:
            continue
        score = (
            sub[[c for c in ["n_b_cells", "n_macrophages", "ripley_excess_auc", "paircorr_excess_auc"] if c in sub.columns]]
            .fillna(0)
            .apply(lambda c: (c - c.mean()) / (c.std(ddof=0) + 1e-8))
            .sum(axis=1)
        )
        reps.append(metrics_df.loc[score.idxmax(), ["image", "group", "path"]].to_dict())
    return reps

def run_validation_scoring(out_dir):
    patch_dir = os.path.join(out_dir, "validation_patches")
    if not os.path.isdir(patch_dir):
        return

    files = sorted(glob.glob(os.path.join(patch_dir, "*_validation_manifest_reviewed.csv")))
    if len(files) == 0:
        return

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["manual_label", "predicted_label"])
    df = df[(df["manual_label"].astype(str).str.strip() != "")]

    if df.empty:
        return

    labels = ["B_cell", "Macrophage", "Other", "DoublePositive"]
    cm = confusion_matrix(df["manual_label"], df["predicted_label"], labels=labels)
    prec, rec, f1, sup = precision_recall_fscore_support(
        df["manual_label"], df["predicted_label"], labels=labels, zero_division=0
    )

    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    cm_df.to_csv(os.path.join(out_dir, "tables", "validation_confusion_matrix.csv"))

    metrics_df = pd.DataFrame({
        "label": labels,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": sup,
    })
    metrics_df.to_csv(os.path.join(out_dir, "tables", "validation_classification_report.csv"), index=False)
    
    
def main():
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "figures"))
    ensure_dir(os.path.join(OUT_DIR, "tables"))
    ensure_dir(os.path.join(OUT_DIR, "per_image"))
    rng = np.random.default_rng(RNG_SEED)

    files = list_images(ROOT_DIR)
    if len(files) == 0:
        raise SystemExit(f"No images found under {ROOT_DIR}")

    summary_rows, nn_parts, nn_cell_parts, curve_parts, cell_parts = [], [], [], [], []
    dist_store = {"B_to_M": {}, "B_to_B": {}, "M_to_M": {}}

    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {Path(path).name}")
        summary, nn_df, curves_df, cell_df, dist_payload = analyze_one_image(path, OUT_DIR, rng)

        n_total = len(cell_df)
        n_b = int(np.sum(cell_df["cell_type"] == "B_cell")) if not cell_df.empty else 0
        n_m = int(np.sum(cell_df["cell_type"].isin(MACROPHAGE_SPATIAL_TYPES))) if not cell_df.empty else 0

        summary["passes_min_total_cells"] = int(n_total >= MIN_TOTAL_CELLS)
        summary["passes_min_b_cells"] = int(n_b >= MIN_B_CELLS)
        summary["passes_min_macrophages"] = int(n_m >= MIN_M_CELLS)
        summary["passes_all_filters"] = int(
            (n_total >= MIN_TOTAL_CELLS) and
            (n_b >= MIN_B_CELLS) and
            (n_m >= MIN_M_CELLS)
        )

        print(
            "FILTER CHECK:",
            Path(path).name,
            "model=", SEGMENTATION_MODEL,
            "total_cells=", n_total,
            "B_cells=", n_b,
            "macrophages_spatial=", n_m,
            "passes_all=", summary["passes_all_filters"]
        )

        summary_rows.append(summary)

        if not nn_df.empty:
            nn_parts.append(nn_df)

        if not curves_df.empty:
            curve_parts.append(curves_df)

        if not cell_df.empty:
            cell_parts.append(cell_df)

        for k, arr in dist_payload.items():
            if len(arr) > 0:
                arr = np.asarray(arr, dtype=float)
                dist_store[k][(summary["image"], summary["group"])] = arr
                nn_cell_parts.append(pd.DataFrame({
                    "image": summary["image"],
                    "group": summary["group"],
                    "segmentation_model": SEGMENTATION_MODEL,
                    "pair": k,
                    "distance": arr,
                }))

    metrics_df = pd.DataFrame(summary_rows)
    if metrics_df.empty:
        print("No images passed filters.")
        return

    nn_long = pd.concat(nn_parts, ignore_index=True) if nn_parts else pd.DataFrame()
    nn_cell_long = pd.concat(nn_cell_parts, ignore_index=True) if nn_cell_parts else pd.DataFrame(
        columns=["image", "group", "segmentation_model", "pair", "distance"]
    )
    curves_all = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()
    cell_all = pd.concat(cell_parts, ignore_index=True) if cell_parts else pd.DataFrame()

    metrics_df.to_csv(os.path.join(OUT_DIR, "tables", "per_image_spatial_metrics.csv"), index=False)
    nn_long.to_csv(os.path.join(OUT_DIR, "tables", "per_image_nn_summary.csv"), index=False)
    nn_cell_long.to_csv(os.path.join(OUT_DIR, "tables", "cell_level_nn_distances.csv"), index=False)
    curves_all.to_csv(os.path.join(OUT_DIR, "tables", "per_image_curves_long.csv"), index=False)
    cell_all.to_csv(os.path.join(OUT_DIR, "tables", "all_cell_calls.csv"), index=False)

    emd_parts = []
    bins = np.linspace(0, max(120, float(metrics_df[[c for c in ["mean_b_to_m", "mean_b_to_b", "mean_m_to_m"] if c in metrics_df.columns]].max().max() * 3 if not metrics_df.empty else 120)), NN_BINS + 1)
    for pair in ["B_to_M", "B_to_B", "M_to_M"]:
        if dist_store[pair]:
            tmp = compute_template_emd(dist_store[pair], bins)
            tmp = tmp.rename(columns={
                "emd_to_cancer_template": f"{pair}_emd_to_cancer_template",
                "emd_to_normal_template": f"{pair}_emd_to_normal_template",
            })
            emd_parts.append(tmp)
    if emd_parts:
        emd_df = emd_parts[0]
        for nxt in emd_parts[1:]:
            emd_df = emd_df.merge(nxt, on=["image", "group"], how="outer")
        metrics_df = metrics_df.merge(emd_df, on=["image", "group"], how="left")
        metrics_df.to_csv(os.path.join(OUT_DIR, "tables", "per_image_spatial_metrics.csv"), index=False)

    stats_df = summarize_group_tests(metrics_df)
    stats_df.to_csv(os.path.join(OUT_DIR, "tables", "group_level_statistical_tests.csv"), index=False)
    run_validation_scoring(OUT_DIR)
    summarize_detection_vs_manual(OUT_DIR)
    reps = choose_representatives(metrics_df)
    if reps:
        save_representative_panel(reps, os.path.join(OUT_DIR, "figures", "Figure1_representative_images.png"))
        rep = reps[0]
        rep_img = load_image(rep["path"])
        rep_cells = cell_all[cell_all["image"] == rep["image"]]
        rep_lbl, _ = segment_nuclei(rep_img)
        save_qc_figure(rep_img, rep_lbl, rep_cells, os.path.join(OUT_DIR, "figures", "Figure2_qc_segmentation.png"), f"Cell calling QC: {rep['image']}")

    k_only = curves_all[curves_all["metric"] == "ripley_k"].copy()
    g_only = curves_all[curves_all["metric"] == "pair_correlation"].copy()
    if not k_only.empty:
        plot_curve_with_envelope(k_only, "ripley_k", os.path.join(OUT_DIR, "figures", "Figure3_ripley_k.png"), "Cross-type Ripley K: B cells vs macrophages")
    if not g_only.empty:
        plot_curve_with_envelope2(g_only, "pair_correlation", os.path.join(OUT_DIR, "figures", "Figure4_pair_correlation.png"), "Pair correlation: B cells vs macrophages")
    if not nn_long.empty and not nn_cell_long.empty:
        plot_nn_violin(
            nn_long,
            nn_cell_long,
            os.path.join(OUT_DIR, "figures", "Figure4_nearest_neighbor_violin.png"),
        )
    if not stats_df.empty:
        plot_effect_forest(stats_df, os.path.join(OUT_DIR, "figures", "Figure5_effect_forest.png"))
    heat_metrics = [c for c in [
        "n_b_cells", "n_macrophages", "density_b", "density_m", "mean_b_to_m", "mean_b_to_b", "mean_m_to_m",
        "ripley_excess_auc", "paircorr_excess_auc"
    ] if c in metrics_df.columns]
    if heat_metrics:
        plot_metric_heatmap(metrics_df, heat_metrics, os.path.join(OUT_DIR, "figures", "Figure5_metric_heatmap.png"))

    manifest = {
        "root_dir": ROOT_DIR,
        "out_dir": OUT_DIR,
        "n_images": int(len(files)),
        "segmentation_model": SEGMENTATION_MODEL,
        "n_cancer": int(np.sum(metrics_df["group"] == "Cancer")),
        "n_normal": int(np.sum(metrics_df["group"] == "Normal")),
        "figure_files": sorted([str(p) for p in Path(os.path.join(OUT_DIR, "figures")).glob("*.png")]),
        "table_files": sorted([str(p) for p in Path(os.path.join(OUT_DIR, "tables")).glob("*.csv")]),
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
