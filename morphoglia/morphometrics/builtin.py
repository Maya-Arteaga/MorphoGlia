#morphoglia/morphometrics/builtin.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morphometric extraction module for the MorphoGlia pipeline.

Single source of truth for:
  - Canonical soma mask (distance-transform + global dilation)
  - Whole-cell skeleton (true 1px, no pruning)
  - Branches mask (cell minus soma)
  - Skeleton feature masks (endpoints, junctions, slab points, initial points)
  - All morphometric calculators

The HTML viewer (io/html.py) must ONLY render the masks produced here via
make_qc_artifacts(...) so visual QC always matches the numbers.

Created on Sun Jul 20 13:57:41 2025
@author: juanpablomayaarteaga
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from itertools import combinations

from skimage.morphology import skeletonize
from scipy.ndimage import label as cc_label, generate_binary_structure, distance_transform_edt

import ast


try:  # only for type hints (avoid circular import at runtime)
    from ..core.models import CellRecord  # noqa: F401
except Exception:  # pragma: no cover
    CellRecord = object  # type: ignore

# ---------------------------------------------------------------------
# Globals / types
# ---------------------------------------------------------------------

MorphometricDict = Dict[str, float]

# Canonical soma extraction settings (used everywhere)
DIST_FRACTION: float = 0.7          # threshold as fraction of DT max  #0.24   ASTROCYTES
SOMA_DILATE_ITERS: int = 1          # single source of truth dilation  #4      ASTROCYTES
# 8-connectivity for consistent component counting everywhere
STRUCT8 = generate_binary_structure(2, 2)  # 3x3 with diagonals



# ---- Branch-order distribution helpers ----
BRANCH_PCTS = (12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5)  # compact, shape-preserving

# ---------------------------------------------------------------------
# Helpers (I/O-free, pure numpy/OpenCV)
# ---------------------------------------------------------------------
def _tuple_str(xs) -> str:
    """Format as tuple string for CSV (stable, human-readable)."""
    return str(tuple(xs))

def _percentiles(values: np.ndarray, pcts=BRANCH_PCTS) -> List[float]:
    if values is None or len(values) == 0:
        return [0.0 for _ in pcts]
    return [float(np.percentile(values, p)) for p in pcts]

def _component_areas(mask_0255: np.ndarray) -> np.ndarray:
    """Areas (px) of connected components of a thick mask."""
    m = (mask_0255 > 0).astype(np.uint8)
    if m.max() == 0:
        return np.array([], dtype=float)
    num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return np.array([], dtype=float)
    return stats[1:, cv2.CC_STAT_AREA].astype(float)  # skip background

def _component_aspect_ratios(mask_0255: np.ndarray) -> np.ndarray:
    """
    Rotation-invariant aspect ratio per component (>=1.0).
    Uses minAreaRect on external contours; falls back to axis-aligned box if needed.
    """
    m = (mask_0255 > 0).astype(np.uint8)
    if m.max() == 0:
        return np.array([], dtype=float)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ars: List[float] = []
    for cnt in contours:
        if len(cnt) >= 5:
            (_, _), (w, h), _ = cv2.minAreaRect(cnt)
            if w > 0 and h > 0:
                ars.append(float(max(w, h) / min(w, h)))
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 0 and h > 0:
                ars.append(float(max(w, h) / min(w, h)))
    return np.array(ars, dtype=float)

def _segment_counts_from_bo(bo: "BranchOrderArtifacts") -> Tuple[int, int, int]:
    """
    Count *segments* (not pixels) per class using the segment label map.
    Returns (primary_count, intermediate_count, terminal_count).
    """
    seg_lbl = bo.seg_label_map
    if seg_lbl.size == 0:
        return 0, 0, 0

    all_labels = set(np.unique(seg_lbl)) - {0}
    prim_labels = set(np.unique(seg_lbl[bo.primary_skel_mask > 0])) - {0}
    term_labels = set(np.unique(seg_lbl[bo.terminal_skel_mask > 0])) - {0}
    inter_labels = all_labels - prim_labels - term_labels

    return len(prim_labels), len(inter_labels), len(term_labels)


def _parse_tuple_str(s):
    if not s or s == "None":
        return None
    try:
        x, y = ast.literal_eval(s)
        return int(x), int(y)
    except Exception:
        return None

def add_global_centroids(rec, feats):
    """
    rec must expose the ROI origin. Adapt to your record:
      - rec.x0, rec.y0
      - or rec.bbox = (x0, y0, w, h)
    """
    x0, y0 = (getattr(rec, "x0", None), getattr(rec, "y0", None))
    if x0 is None or y0 is None:
        x0, y0 = rec.bbox[0], rec.bbox[1]

    for local_key, global_key in [
        ("Soma_centroid_local", "Soma_centroid"),
        ("CH_centroid_local",   "CH_centroid"),
    ]:
        t = _parse_tuple_str(feats.pop(local_key, "None"))
        if t is None:
            feats[global_key] = "None"
        else:
            gx, gy = t[0] + x0, t[1] + y0
            feats[global_key] = str((int(gx), int(gy)))
            # (Optional numeric columns)
            feats[global_key + "_x"] = int(gx)
            feats[global_key + "_y"] = int(gy)

def _round(v: float, nd: int = 4) -> float:
    return float(round(v, nd))

def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Ensure the ROI is single-channel grayscale."""
    if img.ndim == 3:
        # assume BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def _binary(img: np.ndarray) -> np.ndarray:
    """Ensure binary uint8 mask in {0,255}."""
    img = _ensure_gray(img)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.max() > 1:
        _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    else:
        bin_img = (img > 0).astype(np.uint8) * 255
    return bin_img

def _largest_contour(bin_img: np.ndarray):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def _box_count_fractal(mask: np.ndarray, sizes: List[int] | None = None) -> float:
    """Box-counting fractal dimension on binary mask (True/False or 0/255)."""
    if sizes is None:
        sizes = [2 ** i for i in range(4, 10)]  # 16..512
    thresh = (mask > 0)
    counts: List[int] = []
    for size in sizes:
        count = 0
        for x in range(0, thresh.shape[0], size):
            for y in range(0, thresh.shape[1], size):
                if np.any(thresh[x:x + size, y:y + size]):
                    count += 1
        counts.append(count)
    if len(counts) < 2 or min(counts) <= 0:
        return 0.0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return float(-coeffs[0])

def _count_components(binary_mask: np.ndarray) -> int:
    """Count connected components in a binary mask (8-connectivity)."""
    _, num = cc_label(binary_mask.astype(bool), structure=STRUCT8)
    return int(num)

def _remove_small_cc(mask_0255: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected components with area < min_px from a 0/255 mask."""
    if min_px <= 1:
        return (mask_0255 > 0).astype(np.uint8) * 255
    lbl, n = cc_label((mask_0255 > 0), structure=STRUCT8)
    if n == 0:
        return np.zeros_like(mask_0255)
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_px
    keep[0] = False
    out = keep[lbl]
    return (out.astype(np.uint8)) * 255


def _soma_mask_from_distance(
    bin_img: np.ndarray,
    frac: float = DIST_FRACTION,
    dilate_iters: int = SOMA_DILATE_ITERS
) -> np.ndarray:
    """
    Estimate soma by thresholding the distance transform of the binary cell mask.
    Then keep largest component and (optionally) dilate. Returns uint8 {0,255}.
    """
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return np.zeros_like(bin_img)
    _, soma = cv2.threshold(dist, frac * dist.max(), 255, cv2.THRESH_BINARY)
    soma = soma.astype(np.uint8)

    # keep largest component
    labels, num = cc_label(soma.astype(bool), structure=STRUCT8)

    if num < 1:
        soma = np.zeros_like(bin_img)
    else:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = sizes.argmax()
        soma = ((labels == largest).astype(np.uint8)) * 255

    # optional dilation (single source of truth for all modules)
    if dilate_iters > 0:
        soma = cv2.dilate(soma, np.ones((3, 3), np.uint8), iterations=dilate_iters)

    return soma

def _centroid_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return centroid (x, y) of largest component in mask; None if empty."""
    cnt = _largest_contour(mask)
    if cnt is None:
        return None
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)


def _branch_tortuosity_stats(a: QCArtifacts):
    """
    Tortuosity per branch segment on skeleton_wo_soma:
      tortuosity = geodesic_length / chord_length
    where segments are connected components after removing junction pixels.
    """
    # segmentize: remove junction nodes, keep endpoints and slabs
    seg_mask = ((a.skeleton_wo_soma > 0) & (a.junctions_mask == 0)).astype(np.uint8)

    # label segments (8-connectivity)
    lbl, n = cc_label(seg_mask.astype(bool), structure=STRUCT8)
    if n == 0:
        vals = np.array([], dtype=float)
    else:
        vals = []
        # endpoints on the *full* skeleton (not only seg_mask)
        ep = (a.endpoints_mask > 0)

        for i in range(1, n + 1):
            comp = (lbl == i)
            # pixels of this segment
            ys, xs = np.where(comp)
            if len(xs) < 2:
                continue

            # endpoints that belong to this component
            ep_comp = comp & ep
            eys, exs = np.where(ep_comp)

            if len(exs) >= 2:
                # pick the two end nodes (if >2, use the farthest pair)
                pts = np.stack([exs, eys], axis=1)
                # farthest pair
                d2 = np.sum((pts[None, :, :] - pts[:, None, :])**2, axis=2)
                i1, i2 = np.unravel_index(np.argmax(d2), d2.shape)
                x1, y1 = pts[i1]
                x2, y2 = pts[i2]
            else:
                # fall back to geometric extremes if endpoints weren’t marked
                x1, y1 = xs[0], ys[0]
                x2, y2 = xs[-1], ys[-1]

            # chord length (Euclidean)
            chord = float(np.hypot(x2 - x1, y2 - y1))
            if chord == 0.0:
                continue

            # geodesic length: count pixels in the component (≈ 8-connected path length)
            geodesic = float(np.count_nonzero(comp))

            vals.append(geodesic / chord)

        vals = np.array(vals, dtype=float)

    # robust MAD (unscaled) and optional Gaussian-scaled MAD
    if vals.size:
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        mad_g = 1.4826 * mad
        tstats = {
            "Tortuosity_values": vals,
            "Tortuosity_mean":   float(vals.mean()),
            "Tortuosity_median": med,
            "Tortuosity_mad":    mad,      # robust, non-parametric
            "Tortuosity_mad_g":  mad_g,    # scaled-to-sigma version (optional)
            "Tortuosity_std":    float(vals.std(ddof=0)),
            "Tortuosity_min":    float(vals.min()),
            "Tortuosity_max":    float(vals.max()),
            "Tortuosity_n":      int(vals.size),
        }
    else:
        tstats = {
            "Tortuosity_values": np.array([], dtype=float),
            "Tortuosity_mean":   0.0,
            "Tortuosity_median": 0.0,
            "Tortuosity_mad":    0.0,
            "Tortuosity_mad_g":  0.0,
            "Tortuosity_std":    0.0,
            "Tortuosity_min":    0.0,
            "Tortuosity_max":    0.0,
            "Tortuosity_n":      0,
        }

    return tstats["Tortuosity_values"], tstats



def make_branch_order_artifacts(a: QCArtifacts, *, secondary_mode: str = "rest") -> BranchOrderArtifacts:
    """
    Build a robust 3-way partition of branch *area* (cell−soma):
      - primary      : nearest skeleton pixel belongs to an order-1 segment
      - terminal     : nearest skeleton pixel belongs to a terminal segment
                       (endpoint → junction), not touching soma
      - intermediate : everything else (including junction-adjacent segments)

    The three thick masks are a disjoint partition of a.branches.
    """
    h, w = a.bin_cell.shape[:2]
    zeros_u8  = lambda: np.zeros((h, w), np.uint8)
    zeros_i32 = lambda: np.zeros((h, w), np.int32)

    # Segmentize skeleton (remove junction nodes)
    seg_mask = ((a.skeleton_wo_soma > 0) & (a.junctions_mask == 0)).astype(np.uint8)
    seg_lbl, nseg = cc_label(seg_mask.astype(bool), structure=STRUCT8)

    if nseg == 0:
        return BranchOrderArtifacts(
            primary_skel_mask=zeros_u8(), intermediate_skel_mask=zeros_u8(), terminal_skel_mask=zeros_u8(),
            primary_area_mask=zeros_u8(), intermediate_area_mask=zeros_u8(), terminal_area_mask=zeros_u8(),
            primary_area_px=0, intermediate_area_px=0, terminal_area_px=0,
            seg_label_map=zeros_i32(), seg_order_map=zeros_u8(),
        )

    # Primary seeds: segments containing an initial endpoint (soma-touching endpoint)
    init = (a.initial_points_mask > 0)
    prim_labels = set(np.unique(seg_lbl[init])) - {0}

    # Adjacency via junction pixels
    from itertools import combinations
    from collections import deque
    adj = {i: set() for i in range(1, nseg + 1)}
    jy, jx = np.where(a.junctions_mask > 0)
    for y, x in zip(jy, jx):
        nb = np.unique(seg_lbl[max(y-1,0):y+2, max(x-1,0):x+2])
        nb = [int(v) for v in nb if v != 0]
        for u, v in combinations(set(nb), 2):
            adj[u].add(v); adj[v].add(u)

    # BFS order: 1=primary, 2=intermediate, 3+=tertiary...
    order = {i: 0 for i in range(1, nseg + 1)}
    q = deque()
    for p in prim_labels:
        order[p] = 1
        q.append(p)
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if order[nb] == 0:
                order[nb] = order[cur] + 1
                q.append(nb)

    # order map on skeleton pixels (0 elsewhere)
    seg_order_map = np.zeros_like(seg_lbl, dtype=np.uint8)
    for i, k in order.items():
        if k > 0:
            seg_order_map[seg_lbl == i] = np.uint8(k)

    # Terminal segments (endpoint→junction, not initial)
    ep    = (a.endpoints_mask > 0)
    initp = (a.initial_points_mask > 0)
    junc  = (a.junctions_mask > 0)
    k3 = np.ones((3, 3), np.uint8)

    term_seg_bool = np.zeros_like(seg_lbl, dtype=bool)
    for i in range(1, nseg + 1):
        comp = (seg_lbl == i)
        if not comp.any():
            continue
        if np.count_nonzero(comp & ep) < 1:  # needs a distal endpoint
            continue
        if np.any(comp & initp):             # not an initial endpoint
            continue
        comp_dil = cv2.dilate(comp.astype(np.uint8), k3, iterations=1).astype(bool)
        if not np.any(comp_dil & junc):      # must be next to a junction
            continue
        term_seg_bool |= comp

    # Skeleton class map (covers all skeleton pixels)
    all_skel = (a.skeleton_wo_soma > 0)
    cat_map = np.zeros_like(seg_lbl, dtype=np.uint8)  # 0=bg, 1=primary, 2=intermediate, 3=terminal
    cat_map[all_skel] = 2                              # default to intermediate
    cat_map[(seg_order_map == 1)] = 1                  # primary
    cat_map[term_seg_bool]        = 3                  # terminal override

    # thin masks
    prim_skel = ((seg_order_map == 1).astype(np.uint8)) * 255
    term_skel = (term_seg_bool.astype(np.uint8)) * 255
    inter_skel  = (all_skel.astype(np.uint8)) * 255
    inter_skel[(prim_skel > 0) | (term_skel > 0)] = 0

    # thick masks by nearest skeleton class
    branches = (a.branches > 0)
    primary_area_mask      = zeros_u8()
    intermediate_area_mask = zeros_u8()
    terminal_area_mask     = zeros_u8()

    by, bx = np.where(branches)
    if by.size:
        _, (ii, jj) = distance_transform_edt(~all_skel, return_indices=True)
        c = cat_map[ii[by, bx], jj[by, bx]]
        pick_p = (c == 1)
        pick_t = (c == 3)
        pick_i = ~(pick_p | pick_t)  # intermediate = rest
        primary_area_mask[by[pick_p], bx[pick_p]]        = 255
        terminal_area_mask[by[pick_t], bx[pick_t]]       = 255
        intermediate_area_mask[by[pick_i], bx[pick_i]]   = 255

    return BranchOrderArtifacts(
        primary_skel_mask=prim_skel,
        intermediate_skel_mask=inter_skel,
        terminal_skel_mask=term_skel,
        primary_area_mask=primary_area_mask,
        intermediate_area_mask=intermediate_area_mask,
        terminal_area_mask=terminal_area_mask,
        primary_area_px=int(np.count_nonzero(primary_area_mask)),
        intermediate_area_px=int(np.count_nonzero(intermediate_area_mask)),
        terminal_area_px=int(np.count_nonzero(terminal_area_mask)),
        seg_label_map=seg_lbl.astype(np.int32),
        seg_order_map=seg_order_map,
    )








def make_sholl_artifacts(
    roi: np.ndarray,
    params: "ShollParams" = None,
    *,
    soma_frac: float = DIST_FRACTION,
    soma_dilate_iters: int = SOMA_DILATE_ITERS,
) -> ShollArtifacts:
    """
    Build Sholl masks and counts using:
      - centroid from DT-soma (same as everywhere)
      - 1-px skeleton of whole cell, then remove soma
      - blue circles (perimeter mask) drawn at native size (no padding)
      - green overlaps = skeleton_wo_soma ∧ circle_mask
    """
    if params is None:
        params = ShollParams()

    bin_img = _binary(roi)
    soma = _soma_mask_from_distance(bin_img, frac=soma_frac, dilate_iters=soma_dilate_iters)
    centroid = _centroid_from_mask(soma)
    h, w = bin_img.shape[:2]

    if centroid is None:
        return ShollArtifacts(
            circle_mask=np.zeros((h, w), np.uint8),
            overlap_mask=np.zeros((h, w), np.uint8),
            centroid=None,
            max_distance=0.0,
            num_circles=0,
            crossings=0,
            ring_counts=[],
            ring_radii=[],
        )

    cx, cy = centroid

    skeleton_full = _skeletonize_full(bin_img)
    skeleton_wo_soma = ((skeleton_full > 0) & (soma == 0)).astype(np.uint8) * 255

    max_distance = max(
        np.linalg.norm([cx, cy]),
        np.linalg.norm([cx, h - cy]),
        np.linalg.norm([w - cx, cy]),
        np.linalg.norm([w - cx, h - cy]),
    )

    circle_mask = np.zeros_like(bin_img, dtype=np.uint8)
    overlap_union = np.zeros_like(bin_img, dtype=np.uint8)

    ring_counts: List[int] = []
    ring_radii:  List[int] = []

    for r in range(params.start_radius, int(max_distance), params.step):
        ring = np.zeros_like(bin_img, dtype=np.uint8)
        cv2.circle(ring, (cx, cy), r, 255, 1, lineType=cv2.LINE_8)  # 1-px ring
        circle_mask |= ring

        ov = ((ring > 0) & (skeleton_wo_soma > 0)).astype(np.uint8) * 255
        overlap_union |= ov

        _, n = cc_label(ov.astype(bool), structure=STRUCT8)  # OBJECTS per ring
        ring_counts.append(int(n))
        ring_radii.append(int(r))

    return ShollArtifacts(
        circle_mask=circle_mask,
        overlap_mask=overlap_union,
        centroid=(cx, cy),
        max_distance=float(max_distance),
        num_circles=len(ring_counts),
        crossings=int(sum(ring_counts)),
        ring_counts=ring_counts,
        ring_radii=ring_radii,
    )


def _aspect_ratio_from_mask(mask_0255: np.ndarray) -> float:
    """Aspect ratio (>=1.0) from min-area rectangle of mask; 0.0 if empty."""
    cnt = _largest_contour(mask_0255)
    if cnt is None:
        return 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
    if w == 0 or h == 0:
        return 0.0
    return float(max(w, h) / min(w, h))




# ------------------ Skeleton helpers (no pruning) ---------------------

def _skeletonize_full(bin_img: np.ndarray) -> np.ndarray:
    """
    True 1-pixel skeleton of the WHOLE thresholded cell (no pruning).
    Returns uint8 {0,255}.
    """
    skel_bool = skeletonize((bin_img > 0).astype(bool))
    return (skel_bool.astype(np.uint8)) * 255

def _neighbor_degree_map(skel_bin01: np.ndarray) -> np.ndarray:
    """
    Exact 8-neighborhood degree for skeleton pixels, center excluded.

    This uses explicit integer NumPy addition instead of cv2.filter2D.
    The operation is a discrete topological count and must be bit-for-bit
    reproducible across CPU architectures and OpenCV builds.

    Parameters
    ----------
    skel_bin01
        2-D binary skeleton. Any non-zero value is treated as foreground.

    Returns
    -------
    np.ndarray
        int8 array with neighbor counts in [0, 8].
    """
    s = (np.asarray(skel_bin01) > 0).astype(
        np.int16,
        copy=False,
    )

    if s.ndim != 2:
        raise ValueError(
            "_neighbor_degree_map expects a 2-D skeleton array."
        )

    p = np.pad(
        s,
        pad_width=1,
        mode="constant",
        constant_values=0,
    )

    deg = (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    )

    return deg.astype(
        np.int8,
        copy=False,
    )

# ---------------------------------------------------------------------
# QC Artifacts: single source of truth for the HTML viewer
# ---------------------------------------------------------------------

@dataclass
class QCArtifacts:
    bin_cell: np.ndarray              # uint8 0/255
    soma: np.ndarray                  # uint8 0/255 (already dilated per globals)
    branches: np.ndarray              # uint8 0/255 (cell minus soma)
    skeleton_full: np.ndarray         # uint8 0/255 (true 1px skeleton of whole cell)
    skeleton_wo_soma: np.ndarray      # uint8 0/255 (skeleton minus soma)
    degree_map: np.ndarray            # int8 (neighbors count 0..8 on skeleton_wo_soma)
    endpoints_mask: np.ndarray        # uint8 0/255 (deg == 1)
    junctions_mask: np.ndarray        # uint8 0/255 (deg >= 3)
    slab_mask: np.ndarray             # uint8 0/255 (deg == 2)
    initial_points_mask: np.ndarray   # uint8 0/255 (endpoints touching dilated soma)
    centroid: Optional[Tuple[int, int]]

def make_qc_artifacts(
    roi: np.ndarray,
    *,
    soma_frac: float = DIST_FRACTION,
    soma_dilate_iters: int = SOMA_DILATE_ITERS,
) -> QCArtifacts:
    """
    Build all QC layers once (used both by morphometrics and io_html renderer).

    Layers:
      bin_cell             : thresholded ROI (0/255)
      soma                 : DT soma (largest component) + global dilation
      branches             : bin_cell − soma
      skeleton_full        : 1-px skeleton of whole cell
      skeleton_wo_soma     : skeleton_full with soma removed
      degree_map           : 8-neighborhood degree on skeleton_wo_soma
      endpoints_mask       : degree == 1
      junctions_mask       : degree >= 3
      slab_mask            : degree == 2
      initial_points_mask  : endpoints touching (lightly dilated) soma
      centroid             : from soma
    """
    bin_img = _binary(roi)

    # Canonical soma (used everywhere)
    soma = _soma_mask_from_distance(
        bin_img, frac=soma_frac, dilate_iters=soma_dilate_iters
    )

    # Branches = cell − soma
    branches = ((bin_img > 0) & (soma == 0)).astype(np.uint8) * 255

    # Whole-cell skeleton (no pruning)
    skeleton_full = _skeletonize_full(bin_img)

    # Remove soma from skeleton for feature analysis
    skeleton_wo_soma = ((skeleton_full > 0) & (soma == 0)).astype(np.uint8) * 255

    # Degree map and feature masks
    s01 = (skeleton_wo_soma > 0).astype(np.uint8)
    deg = _neighbor_degree_map(s01)
    endpoints_mask  = ((s01 == 1) & (deg == 1)).astype(np.uint8) * 255
    junctions_mask  = ((s01 == 1) & (deg >= 3)).astype(np.uint8) * 255
    slab_mask       = ((s01 == 1) & (deg == 2)).astype(np.uint8) * 255

    # Initial points: endpoints that touch a lightly dilated soma
    soma_touch = cv2.dilate(soma, np.ones((3, 3), np.uint8), iterations=1)
    initial_points_mask = ((endpoints_mask > 0) & (soma_touch > 0)).astype(np.uint8) * 255

    # Soma centroid
    centroid = _centroid_from_mask(soma)

    return QCArtifacts(
        bin_cell=bin_img,
        soma=soma,
        branches=branches,
        skeleton_full=skeleton_full,
        skeleton_wo_soma=skeleton_wo_soma,
        degree_map=deg,
        endpoints_mask=endpoints_mask,
        junctions_mask=junctions_mask,
        slab_mask=slab_mask,
        initial_points_mask=initial_points_mask,
        centroid=centroid,
    )

# ---------------------------------------------------------------------
# Morphometric extractors
# ---------------------------------------------------------------------



@dataclass
class BranchOrderArtifacts:
    # thin (skeleton)
    primary_skel_mask: np.ndarray        # 0/255
    intermediate_skel_mask: np.ndarray   # 0/255  # was "secondary"
    terminal_skel_mask: np.ndarray       # 0/255

    # thick (area on cell−soma)
    primary_area_mask: np.ndarray        # 0/255
    intermediate_area_mask: np.ndarray   # 0/255  # was "secondary"
    terminal_area_mask: np.ndarray       # 0/255

    # pixel counts
    primary_area_px: int
    intermediate_area_px: int            # was "secondary_area_px"
    terminal_area_px: int

    # debugging
    seg_label_map: np.ndarray            # int32
    seg_order_map: np.ndarray            # uint8






@dataclass


class MorphometricExtractor:
    """Base class for morphometric extractors."""
    def __call__(self, roi: np.ndarray) -> MorphometricDict:
        raise NotImplementedError

class CellMorphometrics(MorphometricExtractor):
    """
    Basic shape metrics on the full cell mask:
      - Cell_area, Cell_perimeter, Cell_circularity, Cell_compactness
      - Cell_eccentricity, Cell_aspect_ratio, Cell_feret_diameter, Cell_orientation
    """
    def __call__(self, roi: np.ndarray) -> MorphometricDict:
        bin_img = _binary(roi)
        area = cv2.countNonZero(bin_img)
        cnt = _largest_contour(bin_img)
        if cnt is None or area == 0:
            return {
                "Cell_area": 0.0, "Cell_perimeter": 0.0, "Cell_circularity": 0.0,
                "Cell_compactness": 0.0, "Cell_orientation": 0.0,
                "Cell_feret_diameter": 0.0, "Cell_eccentricity": 0.0,
                "Cell_aspect_ratio": 0.0
            }

        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter else 0.0
        compactness = (area / perimeter) if perimeter else 0.0

        ecc = 0.0
        orientation = 0.0
        feret = 0.0
        aspect_ratio = 0.0

        if len(cnt) >= 5:
            _, (MA, ma), angle = cv2.fitEllipse(cnt)
            ecc = (MA / ma) if ma != 0 else 0.0
            orientation = angle

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = (w / h) if h else 0.0
        rect = cv2.minAreaRect(cnt)
        feret = max(rect[1])

        return {
            "Cell_area": _round(area),
            "Cell_perimeter": _round(perimeter),
            "Cell_circularity": _round(circularity),
            "Cell_compactness": _round(compactness),
            "Cell_orientation": _round(orientation),
            "Cell_feret_diameter": _round(feret),
            "Cell_eccentricity": _round(ecc),
            "Cell_aspect_ratio": _round(aspect_ratio)
        }






class SomaMorphometrics(MorphometricExtractor):
    """
    Distance-transform soma + shape metrics:
      - Soma_area, Soma_perimeter, Soma_circularity, Soma_compactness
      - Soma_orientation, Soma_feret_diameter, Soma_eccentricity, Soma_aspect_ratio
    """
    def __init__(self, dist_fraction: float = DIST_FRACTION):
        self.frac = dist_fraction

    def __call__(self, roi: np.ndarray) -> MorphometricDict:
        bin_img = _binary(roi)
        soma_mask = _soma_mask_from_distance(bin_img, self.frac)
        area = cv2.countNonZero(soma_mask)
        cnt = _largest_contour(soma_mask)
        if cnt is None or area == 0:
            return {
                "Soma_area": 0.0, "Soma_perimeter": 0.0, "Soma_circularity": 0.0,
                "Soma_compactness": 0.0, "Soma_orientation": 0.0,
                "Soma_feret_diameter": 0.0, "Soma_eccentricity": 0.0,
                "Soma_aspect_ratio": 0.0
            }

        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter else 0.0
        compactness = (area / perimeter) if perimeter else 0.0

        ecc = 0.0
        orientation = 0.0
        feret = 0.0
        aspect_ratio = 0.0

        if len(cnt) >= 5:
            _, (MA, ma), angle = cv2.fitEllipse(cnt)
            ecc = (MA / ma) if ma != 0 else 0.0
            orientation = angle

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = (w / h) if h else 0.0
        rect = cv2.minAreaRect(cnt)
        feret = max(rect[1])

        return {
            "Soma_area": _round(area),
            "Soma_perimeter": _round(perimeter),
            "Soma_circularity": _round(circularity),
            "Soma_compactness": _round(compactness),
            "Soma_orientation": _round(orientation),
            "Soma_feret_diameter": _round(feret),
            "Soma_eccentricity": _round(ecc),
            "Soma_aspect_ratio": _round(aspect_ratio)
        }

class ConvexHullMorphometrics(MorphometricExtractor):
    """
    Convex hull metrics on the cell mask.
    Also returns solidity/convexity vs. the original cell.
    """
    def __call__(self, roi: np.ndarray) -> MorphometricDict:
        bin_img = _binary(roi)
        cnt = _largest_contour(bin_img)
        if cnt is None:
            return {
                "Convex_Hull_area": 0.0, "Convex_Hull_perimeter": 0.0,
                "Convex_Hull_compactness": 0.0, "Convex_Hull_eccentricity": 0.0,
                "Convex_Hull_feret_diameter": 0.0, "Convex_Hull_orientation": 0.0,
                "Cell_solidity": 0.0, "Cell_convexity": 0.0,
                "Fractal_dimension": 0.0,
            }

        hull = cv2.convexHull(cnt)
        hull_mask = np.zeros_like(bin_img)
        cv2.fillConvexPoly(hull_mask, hull, 255)
        area = cv2.countNonZero(hull_mask)
        h_cnt = _largest_contour(hull_mask)
        if h_cnt is None:
            return {
                "Convex_Hull_area": 0.0, "Convex_Hull_perimeter": 0.0,
                "Convex_Hull_compactness": 0.0, "Convex_Hull_eccentricity": 0.0,
                "Convex_Hull_feret_diameter": 0.0, "Convex_Hull_orientation": 0.0,
                "Cell_solidity": 0.0, "Cell_convexity": 0.0,
                "Fractal_dimension": 0.0,
            }

        perimeter = cv2.arcLength(h_cnt, True)
        compactness = (area / perimeter) if perimeter else 0.0

        ecc = 0.0
        orientation = 0.0
        feret = 0.0
        if len(h_cnt) >= 5:
            _, (MA, ma), angle = cv2.fitEllipse(h_cnt)
            ecc = (MA / ma) if ma != 0 else 0.0
            orientation = angle
        rect = cv2.minAreaRect(h_cnt)
        feret = max(rect[1])

        cell_area = cv2.countNonZero(bin_img)
        cell_perimeter = cv2.arcLength(cnt, True) if cnt is not None else 0.0
        solidity  = (cell_area / area) if area else 0.0
        convexity = (perimeter / cell_perimeter) if cell_perimeter else 0.0

        return {
            "Convex_Hull_area": _round(area),
            "Convex_Hull_perimeter": _round(perimeter),
            "Convex_Hull_compactness": _round(compactness),
            "Convex_Hull_eccentricity": _round(ecc),
            "Convex_Hull_feret_diameter": _round(feret),
            "Convex_Hull_orientation": _round(orientation),
            "Cell_solidity":  _round(solidity),
            "Cell_convexity": _round(convexity),
            # Keep fractal separate if computed elsewhere; placeholder 0.0 here:
            "Fractal_dimension": 0.0,
        }

class FractalMorphometrics(MorphometricExtractor):
    """Box-counting fractal dimension on binary ROI."""
    def __init__(self, sizes: List[int] | None = None):
        self.sizes = sizes

    def __call__(self, roi: np.ndarray) -> MorphometricDict:
        bin_img = _binary(roi)
        fd = _box_count_fractal(bin_img, self.sizes)
        return {"Fractal_dimension": _round(fd)}






@dataclass
class BranchOrderDistributionMorphometrics(MorphometricExtractor):
    """
    Compact, per-cell distribution descriptor of branch orders.

    For each of Primary / Secondary / Terminal:
      - *_Branch_Count : number of skeleton segments of that order
      - Percentiles_*_Branch_Area : tuple of percentiles of thick-component areas (px)
      - Percentiles_*_Branch_AR   : tuple of percentiles of component aspect ratios (>=1.0)

    Notes:
      • Areas are computed on the thick masks (cell−soma partition), per connected component.
      • Aspect ratios use rotation-invariant minAreaRect when possible; falls back to axis-aligned box.
      • Percentiles default to BRANCH_PCTS = (12.5, 25, 37.5, 50, 62.5, 75, 87.5) for a compact,
        shape-preserving summary suitable for clustering and later violin/KDE reconstruction.
    """
    def __init__(
        self,
        percentiles: Tuple[float, ...] = BRANCH_PCTS,
        dist_fraction: float = DIST_FRACTION,
    ):
        self.percentiles = percentiles
        self.dist_fraction = dist_fraction

    def __call__(self, roi: np.ndarray) -> MorphometricDict:
        a  = make_qc_artifacts(
            roi,
            soma_frac=self.dist_fraction,         # ← use the instance value
            soma_dilate_iters=SOMA_DILATE_ITERS,
        )
        bo = make_branch_order_artifacts(a, secondary_mode="rest")

        # Counts of branch *segments* per class
        nP, nI, nT = _segment_counts_from_bo(bo)

        # Component areas (px) per class
        A_P = _component_areas(bo.primary_area_mask)
        A_I = _component_areas(bo.intermediate_area_mask)
        A_T = _component_areas(bo.terminal_area_mask)

        # Aspect ratios per component
        AR_P = _component_aspect_ratios(bo.primary_area_mask)
        AR_I = _component_aspect_ratios(bo.intermediate_area_mask)
        AR_T = _component_aspect_ratios(bo.terminal_area_mask)

        # Percentiles
        pA_P = _percentiles(A_P, self.percentiles)
        pA_I = _percentiles(A_I, self.percentiles)
        pA_T = _percentiles(A_T, self.percentiles)
        pAR_P = _percentiles(AR_P, self.percentiles)
        pAR_I = _percentiles(AR_I, self.percentiles)
        pAR_T = _percentiles(AR_T, self.percentiles)


        return {
            # Counts
            "Primary_Branch_Count":       float(nP),
            "Intermediate_Branch_Count":  float(nI),
            "Terminal_Branch_Count":      float(nT),
        
            # Area distributions (px) as tuple strings
            "Percentiles_Primary_Branch_Area":      _tuple_str([_round(v) for v in pA_P]),
            "Percentiles_Intermediate_Branch_Area": _tuple_str([_round(v) for v in pA_I]),
            "Percentiles_Terminal_Branch_Area":     _tuple_str([_round(v) for v in pA_T]),
        
            # Aspect ratio distributions (>=1.0)
            "Percentiles_Primary_Branch_AR":        _tuple_str([_round(v) for v in pAR_P]),
            "Percentiles_Intermediate_Branch_AR":   _tuple_str([_round(v) for v in pAR_I]),
            "Percentiles_Terminal_Branch_AR":       _tuple_str([_round(v) for v in pAR_T]),
        }





@dataclass
class ShollParams:
    start_radius: int = 25
    step: int = 10



@dataclass
class ShollArtifacts:
    """Masks & metrics for Sholl analysis (native image size, no padding)."""
    circle_mask: np.ndarray        # uint8 0/255; union of all circle perimeters
    overlap_mask: np.ndarray       # uint8 0/255; union of overlaps with all rings
    centroid: Optional[Tuple[int, int]]
    max_distance: float
    num_circles: int
    crossings: int                 # sum of objects across rings
    ring_counts: List[int]         # NEW: objects per ring (radius order)
    ring_radii: List[int]          # NEW: radii used (same order)

class ShollMorphometrics(MorphometricExtractor):
    """
    Sholl analysis using soma-centroid + whole-cell 1px skeleton (no padding).
    Returns:
      - Sholl_max_distance
      - Sholl_crossing_processes  (connected components of overlaps)
      - Sholl_circles             (number of radii drawn)
    """
    def __init__(self, params: "ShollParams" = None, soma_frac: float = DIST_FRACTION):
        self.params = params if params is not None else ShollParams()
        self.soma_frac = soma_frac

    def __call__(self, roi: np.ndarray) -> MorphometricDict:
        sh = make_sholl_artifacts(
            roi,
            params=self.params,
            soma_frac=self.soma_frac,
            soma_dilate_iters=SOMA_DILATE_ITERS,
        )
        # tuple-as-string so it fits nicely in CSV
        ring_tuple_str = str(tuple(sh.ring_counts))
        radii_tuple_str = str(tuple(sh.ring_radii))
        return {
            "Sholl_max_distance": _round(float(sh.max_distance)),
            "Sholl_crossing_processes": _round(float(sh.crossings)),
            "Sholl_circles": float(sh.num_circles),
            "Sholl_ring_counts": ring_tuple_str,
            "Sholl_ring_radii": radii_tuple_str,   # optional: keeps linters happy
        }





    
class BranchingMorphometrics(MorphometricExtractor):
    """
    Legacy-like skeleton metrics using the SAME soma mask used everywhere else
    and the true 1-px skeleton (no pruning):

      - End_Points            : deg == 1
      - Junctions             : deg >= 3
      - Branches              : connected components where deg == 2
      - Initial_Points        : endpoints that touch (dilated) soma
      - Total_Branches_Length : components(deg == 2) + End_Points
      - ratio_branches        : End_Points / Initial_Points
      - Branches_area         : area(cell − soma)
    """
    def __init__(self, soma_frac: float = DIST_FRACTION):
        self.soma_frac = soma_frac


    def __call__(self, roi: np.ndarray) -> MorphometricDict:
        a = make_qc_artifacts(
            roi,
            soma_frac=self.soma_frac,
            soma_dilate_iters=SOMA_DILATE_ITERS,
        )
    
        def _cc(mask_0255: np.ndarray) -> int:
            _, n = cc_label((mask_0255 > 0), structure=STRUCT8)
            return int(n)
    
        end_points     = _cc(a.endpoints_mask)
        junctions      = _cc(a.junctions_mask)
        seg_mask       = ((a.skeleton_wo_soma > 0) & (a.junctions_mask == 0)).astype(np.uint8) * 255
        branches       = _cc(seg_mask)
        initial_points = _cc(a.initial_points_mask)
        total_len      = int(np.count_nonzero(a.skeleton_wo_soma))
    
        ratio         = (end_points / initial_points) if initial_points else 0.0
        branches_area = int(np.count_nonzero(a.branches))
        cell_area     = int(np.count_nonzero(a.bin_cell))
    
        bo = make_branch_order_artifacts(a, secondary_mode="rest")
    
        # exact 3-way area partition (px)
        P = int(bo.primary_area_px)
        I = int(bo.intermediate_area_px)
        T = int(bo.terminal_area_px)
    
        # Ensure exact partition (rare EDT ties → push to intermediate)
        diff = branches_area - (P + I + T)
        if diff != 0:
            I = max(0, I + diff)
    
        # per-component areas (px) for mean/median
        A_P = _component_areas(bo.primary_area_mask)
        A_I = _component_areas(bo.intermediate_area_mask)
        A_T = _component_areas(bo.terminal_area_mask)
    
        # counts of branch *segments* per order
        nP, nI, nT = _segment_counts_from_bo(bo)
    
        # order-specific skeleton lengths (px) for thickness
        L_P = int(np.count_nonzero(bo.primary_skel_mask))
        L_I = int(np.count_nonzero(bo.intermediate_skel_mask))
        L_T = int(np.count_nonzero(bo.terminal_skel_mask))
    
        # aspect ratios (>=1.0)
        ar_P = _aspect_ratio_from_mask(bo.primary_area_mask)
        ar_I = _aspect_ratio_from_mask(bo.intermediate_area_mask)
        ar_T = _aspect_ratio_from_mask(bo.terminal_area_mask)
    
        # --- NEW metrics ---
        def mean_or_0(x):   return float(np.mean(x))    if x.size else 0.0
        def median_or_0(x): return float(np.median(x))  if x.size else 0.0
        def safe_div(a, b): return (a / b) if b else 0.0
    
        # Area statistics (px)
        Prim_Area_Mean   = mean_or_0(A_P)
        Inter_Area_Mean  = mean_or_0(A_I)
        Term_Area_Mean   = mean_or_0(A_T)
    
        Prim_Area_Median  = median_or_0(A_P)
        Inter_Area_Median = median_or_0(A_I)
        Term_Area_Median  = median_or_0(A_T)
    
        # Fraction of *cell* area
        Prim_Area_Fraction  = safe_div(P, cell_area)
        Inter_Area_Fraction = safe_div(I, cell_area)
        Term_Area_Fraction  = safe_div(T, cell_area)
    
        # Mean per-branch area normalized by cell area
        Prim_Area_NormCell  = safe_div(safe_div(P, max(nP, 1)), cell_area)
        Inter_Area_NormCell = safe_div(safe_div(I, max(nI, 1)), cell_area)
        Term_Area_NormCell  = safe_div(safe_div(T, max(nT, 1)), cell_area)
    
        # Thickness proxy (px): area per unit skeleton length
        Prim_Thickness  = safe_div(P, L_P)
        Inter_Thickness = safe_div(I, L_I)
        Term_Thickness  = safe_div(T, L_T)
    
        # tortuosity (unchanged)
        _, tstats = _branch_tortuosity_stats(a)
    
        return {
            "Branches_area":         _round(float(branches_area)),
            "End_Points":            _round(float(end_points)),
            "Junctions":             _round(float(junctions)),
            "Branches":              _round(float(branches)),
            "Initial_Points":        _round(float(initial_points)),
            "Total_Branches_Length": _round(float(total_len)),
            "ratio_branches":        _round(float(ratio)),
    
            "Tortuosity_median":     _round(float(tstats.get("Tortuosity_median", 0.0))),
            "Tortuosity_mad":        _round(float(tstats.get("Tortuosity_mad",    0.0))),
    
            # exact 3-way areas
            "Primary_Branches_area":      _round(float(P)),
            "Intermediate_Branches_area": _round(float(I)),
            "Terminal_Branches_area":     _round(float(T)),
    
            # aspect ratios
            "Primary_Branches_aspect_ratio":      _round(float(ar_P)),
            "Intermediate_Branches_aspect_ratio": _round(float(ar_I)),
            "Terminal_Branches_aspect_ratio":     _round(float(ar_T)),
    
            # NEW per-order stats
            "Primary_Area_Mean":      _round(Prim_Area_Mean),
            "Intermediate_Area_Mean": _round(Inter_Area_Mean),
            "Terminal_Area_Mean":     _round(Term_Area_Mean),
    
            "Primary_Area_Median":      _round(Prim_Area_Median),
            "Intermediate_Area_Median": _round(Inter_Area_Median),
            "Terminal_Area_Median":     _round(Term_Area_Median),
    
            "Primary_Area_Fraction":      _round(Prim_Area_Fraction),
            "Intermediate_Area_Fraction": _round(Inter_Area_Fraction),
            "Terminal_Area_Fraction":     _round(Term_Area_Fraction),
    
            "Primary_Area_NormCell":      _round(Prim_Area_NormCell),
            "Intermediate_Area_NormCell": _round(Inter_Area_NormCell),
            "Terminal_Area_NormCell":     _round(Term_Area_NormCell),
    
            "Primary_Thickness":      _round(Prim_Thickness),
            "Intermediate_Thickness": _round(Inter_Thickness),
            "Terminal_Thickness":     _round(Term_Thickness),
        }



class CentroidMorphometrics(MorphometricExtractor):
    """
    Returns ROI-local integer centroids as tuple-strings:
      - Soma_centroid_local: (x, y) of DT-based soma (with global dilation)
      - CH_centroid_local  : (x, y) of convex-hull centroid of the cell

    NOTE: These are ROI-local. Convert to global with your crop origin (x0, y0)
    at IO/assembly time to produce:
      - Soma_centroid
      - CH_centroid
    """
    def __init__(self, dist_fraction: float = DIST_FRACTION):
        self.frac = dist_fraction

    def __call__(self, roi: np.ndarray) -> MorphometricDict:
        bin_img = _binary(roi)

        # Soma centroid (same soma mask used everywhere)
        soma_mask = _soma_mask_from_distance(
            bin_img, frac=self.frac, dilate_iters=SOMA_DILATE_ITERS
        )
        soma_c = _centroid_from_mask(soma_mask)  # -> (cx, cy) or None

        # Convex-hull centroid
        ch_c = None
        cnt = _largest_contour(bin_img)
        if cnt is not None and len(cnt) >= 3:
            hull = cv2.convexHull(cnt)
            M = cv2.moments(hull)
            if M["m00"] != 0:
                ch_c = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        return {
            "Soma_centroid_local": _tuple_str(soma_c) if soma_c is not None else "None",
            "CH_centroid_local":   _tuple_str(ch_c)   if ch_c   is not None else "None",
        }

# ---------------------------------------------------------------------
# DESCRIPTIONS
# ---------------------------------------------------------------------


    

    
METRIC_DESCRIPTIONS = {
    # ── Cell (full mask) ─────────────────────────────────────────────────────
    "Cell_area":           "Area (px) of the thresholded full-cell mask (0/255).",
    "Cell_perimeter":      "Perimeter length (px) of the largest external contour of the cell mask.",
    "Cell_circularity":    "4π·area / perimeter² (dimensionless; 1 = perfect circle).",
    "Cell_compactness":    "area / perimeter (px).",
    "Cell_orientation":    "Orientation (deg) of fitEllipse major axis (OpenCV 0–180°) on the cell contour.",
    "Cell_feret_diameter": "Max caliper (px) from minAreaRect (largest rectangle side).",
    "Cell_eccentricity":   "Ellipse axis ratio MA/ma (>=1) from fitEllipse on the cell contour.",
    "Cell_aspect_ratio":   "Axis-aligned bounding-box width / height (>=0).",

    # ── Soma (DT-based + global dilation) ────────────────────────────────────
    "Soma_area":           "Area (px) of DT-thresholded soma (largest component) after global dilation.",
    "Soma_perimeter":      "Perimeter length (px) of the soma mask.",
    "Soma_circularity":    "4π·area / perimeter² of the soma mask.",
    "Soma_compactness":    "Soma area / soma perimeter (px).",
    "Soma_orientation":    "Orientation (deg) of fitEllipse major axis on the soma contour.",
    "Soma_feret_diameter": "Max caliper (px) of the soma from minAreaRect.",
    "Soma_eccentricity":   "Ellipse axis ratio MA/ma (>=1) from fitEllipse on soma.",
    "Soma_aspect_ratio":   "Axis-aligned bounding-box width / height of the soma mask.",

    # ── Convex hull + derived vs cell ────────────────────────────────────────
    "Convex_Hull_area":         "Area (px) of the convex hull of the full-cell mask.",
    "Convex_Hull_perimeter":    "Perimeter (px) of the convex hull contour.",
    "Convex_Hull_compactness":  "Hull area / hull perimeter (px).",
    "Convex_Hull_eccentricity": "Ellipse axis ratio MA/ma (>=1) from fitEllipse on the hull.",
    "Convex_Hull_feret_diameter":"Max caliper (px) from minAreaRect on the hull.",
    "Convex_Hull_orientation":  "Orientation (deg) of fitEllipse major axis on the hull.",
    "Cell_solidity":            "Cell_area / Convex_Hull_area (dimensionless).",
    "Cell_convexity":           "Convex_Hull_perimeter / Cell_perimeter (dimensionless).",
    "Fractal_dimension":        "Box-counting fractal dimension of the cell mask (sizes 16..512 px by default).",

    # ── Branching (using global soma + true 1px skeleton) ────────────────────
    "Branches_area":                 "Area (px) of branches = Cell mask minus (dilated) soma.",
    "End_Points":                    "Count of skeleton pixels with degree == 1 (on skeleton_wo_soma).",
    "Junctions":                     "Count of skeleton pixels with degree >= 3 (on skeleton_wo_soma).",
    "Branches":                      "Count of connected components where deg == 2 after junction removal.",
    "Initial_Points":                "Endpoints touching the (lightly dilated) soma.",
    "Total_Branches_Length":         "Total 1-px length (px) of skeleton_wo_soma.",
    "ratio_branches":                "End_Points / Initial_Points (0 if denominator is 0).",

    # Exact 3-way area partition of branches (primary / intermediate / terminal)
    "Primary_Branches_area":         "Area (px) assigned to PRIMARY branches via nearest-skeleton order.",
    "Intermediate_Branches_area":    "Area (px) assigned to INTERMEDIATE branches (incl. ties).",
    "Terminal_Branches_area":        "Area (px) assigned to TERMINAL endpoint→junction branches.",
    "Primary_Branches_aspect_ratio": "Component aspect ratio (>=1) of PRIMARY thick regions (minAreaRect).",
    "Intermediate_Branches_aspect_ratio":"Component aspect ratio (>=1) of INTERMEDIATE thick regions.",
    "Terminal_Branches_aspect_ratio":"Component aspect ratio (>=1) of TERMINAL thick regions.",

    # Per-order distributions (compact percentiles)
    "Primary_Branch_Count":       "Number of PRIMARY branch segments (after junction removal).",
    "Intermediate_Branch_Count":  "Number of INTERMEDIATE branch segments.",
    "Terminal_Branch_Count":      "Number of TERMINAL branch segments (endpoint→junction, not soma).",
    "Percentiles_Primary_Branch_Area":      f"Percentiles {BRANCH_PCTS} of PRIMARY component areas (px).",
    "Percentiles_Intermediate_Branch_Area": f"Percentiles {BRANCH_PCTS} of INTERMEDIATE component areas (px).",
    "Percentiles_Terminal_Branch_Area":     f"Percentiles {BRANCH_PCTS} of TERMINAL component areas (px).",
    "Percentiles_Primary_Branch_AR":        f"Percentiles {BRANCH_PCTS} of PRIMARY component aspect ratios (>=1).",
    "Percentiles_Intermediate_Branch_AR":   f"Percentiles {BRANCH_PCTS} of INTERMEDIATE component aspect ratios (>=1).",
    "Percentiles_Terminal_Branch_AR":       f"Percentiles {BRANCH_PCTS} of TERMINAL component aspect ratios (>=1).",

    # Per-order summaries
    "Primary_Area_Mean":           "Mean component area (px) of PRIMARY branches.",
    "Intermediate_Area_Mean":      "Mean component area (px) of INTERMEDIATE branches.",
    "Terminal_Area_Mean":          "Mean component area (px) of TERMINAL branches.",
    "Primary_Area_Median":         "Median component area (px) of PRIMARY branches.",
    "Intermediate_Area_Median":    "Median component area (px) of INTERMEDIATE branches.",
    "Terminal_Area_Median":        "Median component area (px) of TERMINAL branches.",
    "Primary_Area_Fraction":       "PRIMARY thick area / Cell_area (dimensionless).",
    "Intermediate_Area_Fraction":  "INTERMEDIATE thick area / Cell_area (dimensionless).",
    "Terminal_Area_Fraction":      "TERMINAL thick area / Cell_area (dimensionless).",
    "Primary_Area_NormCell":       "Mean per-branch PRIMARY area / Cell_area (dimensionless).",
    "Intermediate_Area_NormCell":  "Mean per-branch INTERMEDIATE area / Cell_area (dimensionless).",
    "Terminal_Area_NormCell":      "Mean per-branch TERMINAL area / Cell_area (dimensionless).",
    "Primary_Thickness":           "PRIMARY area / PRIMARY skeleton length (px).",
    "Intermediate_Thickness":      "INTERMEDIATE area / INTERMEDIATE skeleton length (px).",
    "Terminal_Thickness":          "TERMINAL area / TERMINAL skeleton length (px).",

    # Tortuosity (per-segment; robust summary)
    "Tortuosity_median":           "Median geodesic/chord length ratio over segments (skeleton_wo_soma).",
    "Tortuosity_mad":              "Median absolute deviation of tortuosity (robust).",

    # Sholl
    "Sholl_max_distance":          "Max radius (px) swept from soma centroid to image corners.",
    "Sholl_crossing_processes":    "Total count of overlap components across all rings (sum over radii).",
    "Sholl_circles":               "Number of radii drawn in Sholl analysis.",
    "Sholl_ring_counts":           "Tuple of overlap-component counts per radius (native image scale).",
    "Sholl_ring_radii":            "Tuple of radii (px) used in Sholl analysis.",

    # Centroids (local and global)
    "Soma_centroid_local":         "ROI-local (x, y) centroid of the DT-based (dilated) soma mask.",
    "CH_centroid_local":           "ROI-local (x, y) centroid of the convex hull of the cell mask.",
    "Soma_centroid":               "GLOBAL (x, y) soma centroid = local + (bbox_x, bbox_y).",
    "Soma_centroid_x":             "GLOBAL soma centroid x (px).",
    "Soma_centroid_y":             "GLOBAL soma centroid y (px).",
    "CH_centroid":                 "GLOBAL (x, y) convex-hull centroid = local + (bbox_x, bbox_y).",
    "CH_centroid_x":               "GLOBAL convex-hull centroid x (px).",
    "CH_centroid_y":               "GLOBAL convex-hull centroid y (px).",
}