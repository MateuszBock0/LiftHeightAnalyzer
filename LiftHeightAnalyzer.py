import csv
import os
import queue
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# APPLICATION / PATH SETTINGS
# ============================================================

def application_folder():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_FOLDER = application_folder()
DEFAULT_EXPERIMENTS_FOLDER = APP_FOLDER / "Experiments"
OUTPUT_ROOT_NAME = "_analysis"


# ============================================================
# VIDEO OUTPUT
# ============================================================

WRITE_MARKED_VIDEO = True
VIDEO_CODEC = "mp4v"


# ============================================================
# PHYSICAL SCALE
# ============================================================

# One ruler division = 1 mm
MM_PER_RULER_UNIT = 1.0

# Highest visible ruler value.
# 79 is a minor mark, not a complete major mark.
TOP_VISIBLE_VALUE = 79

# First complete major line.
TOP_MAJOR_VALUE = 75

# Lowest detected major line used as an anchor.
BOTTOM_MAJOR_VALUE = 5

# Major line spacing in ruler units.
MAJOR_STEP = 5

TOP_MINOR_REFERENCE_INTERVALS = 3
BOTTOM_REFERENCE_INTERVALS = 4


# ============================================================
# DIAGNOSTIC SMOOTHING
# ============================================================

# Only used for the diagnostic dashed curve.
# It is NOT used for folder averages.
TEMPORAL_WINDOW = 5


# ============================================================
# MONOTONIC TRACKING
# ============================================================

# A newly confirmed level must exceed the previously confirmed
# level by at least this amount.
TRACK_MIN_RISE_MM = 0.02

# Detection-quality threshold for a frame to participate in
# confirmation. This percentage is an intuitive display metric,
# not a calibrated probability of correctness.
TRACK_MIN_CONFIDENCE_PERCENT = 55.0

# Confirmation rule requested:
# at least 3 of 5 consecutive frames must support roughly the
# same new level.
TRACK_CONFIRM_WINDOW = 5
TRACK_CONFIRM_REQUIRED = 3
TRACK_CONFIRM_TOLERANCE_MM = 0.30

# Interpolate between confirmed anchors to obtain the smooth
# purple-type physical reconstruction.
TRACK_INTERPOLATE_BETWEEN_ANCHORS = True


# ============================================================
# RULER DETECTION
# ============================================================

RULER_DARK_THRESHOLD = 110
RULER_MIN_DARK_FRACTION = 0.40
RULER_BOX_PAD_REL = 0.03


# ============================================================
# MAJOR RULER MARK DETECTION
# ============================================================

MAJOR_MIN_DISTANCE_REL = 0.028
MAJOR_PAIR_TOLERANCE_REL = 0.02


# ============================================================
# SAMPLE DETECTION
# ============================================================

SAMPLE_SEARCH_LEFT = 0.35
SAMPLE_SEARCH_RIGHT = 0.65

SAMPLE_WIDTH_MIN_RATIO = 0.12
SAMPLE_WIDTH_MAX_RATIO = 0.75

SAMPLE_EDGE_ROW_STEP_REL = 0.0015
SAMPLE_EDGE_BLUR_X_REL = 0.005
SAMPLE_EDGE_BLUR_Y_REL = 0.002

# Use only the central part of the porous strip for front
# detection, avoiding reflective edges.
SAMPLE_INNER_MARGIN = 0.20


# ============================================================
# LIQUID FRONT
# ============================================================

FRONT_SMOOTH_REL = 0.035
FRONT_SEARCH_TOP = 0.05
FRONT_SEARCH_BOTTOM = 0.97


# ============================================================
# CONFIDENCE DISPLAY
# ============================================================

# Converts the robust z-like edge-strength score to an intuitive
# 0-100% display value.
#
# It is NOT a statistically calibrated probability.
CONFIDENCE_PERCENT_SCALE = 5.0


# ============================================================
# PLOTTING
# ============================================================

FIGURE_DPI = 200

# Folder averages are calculated over the time interval present
# in every successful video in that folder.
AVERAGE_USE_COMMON_TIME_ONLY = True


# ============================================================
# CALLBACK HELPERS
# ============================================================

def emit_log(callback, message):
    if callback is not None:
        callback(str(message))


def emit_progress(callback, **payload):
    if callback is not None:
        callback(payload)


# ============================================================
# GENERAL HELPERS
# ============================================================

def odd_number(value, minimum=3):
    value = max(minimum, int(round(value)))
    if value % 2 == 0:
        value += 1
    return value


def smooth_1d(signal, kernel_size):
    signal = np.asarray(signal, dtype=np.float32)

    if len(signal) < 3:
        return signal.copy()

    kernel_size = odd_number(kernel_size)

    if kernel_size >= len(signal):
        kernel_size = len(signal) if len(signal) % 2 == 1 else len(signal) - 1

    kernel_size = max(3, kernel_size)

    result = cv2.GaussianBlur(
        signal.reshape(-1, 1),
        (1, kernel_size),
        0
    )

    return result.ravel()


def find_peaks_1d(signal, min_distance, min_value):
    signal = np.asarray(signal, dtype=np.float32)
    candidates = []

    for i in range(1, len(signal) - 1):
        if (
            signal[i] >= signal[i - 1]
            and signal[i] >= signal[i + 1]
            and signal[i] >= min_value
        ):
            candidates.append((i, float(signal[i])))

    candidates.sort(key=lambda item: item[1], reverse=True)

    selected = []

    for index, strength in candidates:
        if all(
            abs(index - previous) >= min_distance
            for previous in selected
        ):
            selected.append(index)

    selected.sort()
    return selected


def centered_median_filter(values, window_size):
    values = np.asarray(values, dtype=np.float64)
    output = values.copy()

    if window_size <= 1:
        return output

    if window_size % 2 == 0:
        window_size += 1

    if len(values) < window_size:
        return output

    half = window_size // 2

    # Edge frames remain raw: no causal lag and no fake padding.
    for i in range(half, len(values) - half):
        output[i] = np.median(
            values[i - half:i + half + 1]
        )

    return output


def confidence_score_to_percent(score):
    if not np.isfinite(score):
        return 0.0

    score = max(0.0, float(score))

    percentage = (
        100.0
        * (
            1.0
            - np.exp(
                -score / CONFIDENCE_PERCENT_SCALE
            )
        )
    )

    return float(
        np.clip(
            percentage,
            0.0,
            100.0
        )
    )


# ============================================================
# 3-OF-5 CONFIRMATION
# ============================================================

def find_confirmed_level_in_window(
    heights,
    confidences,
    global_indices,
    current_height
):
    """
    Find the highest new level supported by at least
    TRACK_CONFIRM_REQUIRED frames within a consecutive
    TRACK_CONFIRM_WINDOW-frame window.

    Supporting frames:
      - meet the confidence threshold;
      - lie within +/- TRACK_CONFIRM_TOLERANCE_MM of the
        candidate cluster.

    Returns None or a dict describing the confirmed cluster.
    """

    heights = np.asarray(heights, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)
    global_indices = np.asarray(global_indices, dtype=int)

    valid = (
        np.isfinite(heights)
        & np.isfinite(confidences)
        & (confidences >= TRACK_MIN_CONFIDENCE_PERCENT)
    )

    if np.sum(valid) < TRACK_CONFIRM_REQUIRED:
        return None

    valid_heights = heights[valid]
    valid_indices = global_indices[valid]

    possible_clusters = []

    for candidate_height in valid_heights:
        support = (
            np.abs(valid_heights - candidate_height)
            <= TRACK_CONFIRM_TOLERANCE_MM
        )

        support_count = int(np.sum(support))

        if support_count < TRACK_CONFIRM_REQUIRED:
            continue

        cluster_heights = valid_heights[support]
        cluster_indices = valid_indices[support]

        confirmed_height = float(np.median(cluster_heights))

        if confirmed_height < current_height + TRACK_MIN_RISE_MM:
            continue

        # Representative anchor time is the earliest supporting
        # measurement nearest to the cluster median.
        distances = np.abs(cluster_heights - confirmed_height)
        best_distance = float(np.min(distances))

        representative_candidates = cluster_indices[
            distances <= best_distance + 1e-9
        ]

        anchor_index = int(np.min(representative_candidates))

        possible_clusters.append(
            {
                "height": confirmed_height,
                "anchor_index": anchor_index,
                "support_indices": cluster_indices.copy(),
                "support_count": support_count,
            }
        )

    if not possible_clusters:
        return None

    # Prefer the highest confirmed cluster, then greater support,
    # then the earliest representative frame.
    return max(
        possible_clusters,
        key=lambda item: (
            item["height"],
            item["support_count"],
            -item["anchor_index"],
        )
    )


def build_monotonic_tracked_curve(
    time_s,
    raw_height_mm,
    confidence_percent
):
    """
    Physically constrained capillary-front reconstruction.

    Safeguards:
      1. New level must be above the current confirmed level.
      2. Supporting frames must meet the confidence threshold.
      3. At least 3 of 5 consecutive frames must agree within
         the configured height tolerance.
      4. Once confirmed, the tracked front cannot decrease.

    Between confirmed anchors, the curve is linearly
    interpolated when TRACK_INTERPOLATE_BETWEEN_ANCHORS=True.
    """

    time_s = np.asarray(time_s, dtype=np.float64)
    raw_height_mm = np.asarray(raw_height_mm, dtype=np.float64)
    confidence_percent = np.asarray(
        confidence_percent,
        dtype=np.float64
    )

    n = len(raw_height_mm)

    if n == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=bool),
            [],
        )

    if not (
        len(time_s) == n
        and len(confidence_percent) == n
    ):
        raise ValueError(
            "Tracking arrays must have equal length."
        )

    window_size = int(TRACK_CONFIRM_WINDOW)
    required = int(TRACK_CONFIRM_REQUIRED)

    if window_size < 1:
        raise ValueError(
            "TRACK_CONFIRM_WINDOW must be >= 1."
        )

    if required < 1:
        raise ValueError(
            "TRACK_CONFIRM_REQUIRED must be >= 1."
        )

    if required > window_size:
        raise ValueError(
            "TRACK_CONFIRM_REQUIRED cannot exceed "
            "TRACK_CONFIRM_WINDOW."
        )

    anchor_indices = [0]
    anchor_heights = [float(raw_height_mm[0])]

    confirmation_records = [
        {
            "anchor_index": 0,
            "height": float(raw_height_mm[0]),
            "support_indices": np.array([0], dtype=int),
            "support_count": 1,
        }
    ]

    current_height = float(raw_height_mm[0])

    last_start = n - window_size

    if last_start >= 0:
        for start in range(0, last_start + 1):
            end = start + window_size

            confirmed = find_confirmed_level_in_window(
                raw_height_mm[start:end],
                confidence_percent[start:end],
                np.arange(start, end, dtype=int),
                current_height,
            )

            if confirmed is None:
                continue

            confirmed_height = float(
                confirmed["height"]
            )

            anchor_index = int(
                confirmed["anchor_index"]
            )

            if anchor_index <= anchor_indices[-1]:
                continue

            if (
                confirmed_height
                < current_height + TRACK_MIN_RISE_MM
            ):
                continue

            anchor_indices.append(anchor_index)
            anchor_heights.append(confirmed_height)
            confirmation_records.append(confirmed)

            current_height = confirmed_height

    anchor_indices = np.asarray(
        anchor_indices,
        dtype=int
    )

    anchor_heights = np.asarray(
        anchor_heights,
        dtype=np.float64
    )

    order = np.argsort(anchor_indices)

    anchor_indices = anchor_indices[order]
    anchor_heights = anchor_heights[order]

    # Remove duplicate times, retaining the highest accepted
    # height at each anchor index.
    unique_indices = []
    unique_heights = []

    for index, height in zip(
        anchor_indices,
        anchor_heights
    ):
        index = int(index)
        height = float(height)

        if (
            unique_indices
            and index == unique_indices[-1]
        ):
            unique_heights[-1] = max(
                unique_heights[-1],
                height
            )
        else:
            unique_indices.append(index)
            unique_heights.append(height)

    anchor_indices = np.asarray(
        unique_indices,
        dtype=int
    )

    anchor_heights = np.asarray(
        unique_heights,
        dtype=np.float64
    )

    anchor_heights = np.maximum.accumulate(
        anchor_heights
    )

    anchor_mask = np.zeros(
        n,
        dtype=bool
    )
    anchor_mask[anchor_indices] = True

    if (
        TRACK_INTERPOLATE_BETWEEN_ANCHORS
        and len(anchor_indices) >= 2
    ):
        tracked = np.interp(
            time_s,
            time_s[anchor_indices],
            anchor_heights,
        )
    else:
        tracked = np.empty(
            n,
            dtype=np.float64
        )

        pointer = 0
        current = float(anchor_heights[0])

        for i in range(n):
            while (
                pointer + 1 < len(anchor_indices)
                and i >= anchor_indices[pointer + 1]
            ):
                pointer += 1
                current = float(
                    anchor_heights[pointer]
                )

            tracked[i] = current

    tracked = np.maximum.accumulate(
        tracked
    )

    return (
        tracked,
        anchor_mask,
        confirmation_records,
    )


# ============================================================
# RULER DETECTION
# ============================================================

def detect_rulers(image):
    h, w = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    y0 = int(0.08 * h)
    y1 = int(0.97 * h)

    band = gray[y0:y1, :]

    dark_fraction = np.mean(
        band < RULER_DARK_THRESHOLD,
        axis=0
    )

    column_mask = (
        dark_fraction
        > RULER_MIN_DARK_FRACTION
    ).astype(np.uint8) * 255

    close_width = odd_number(
        0.018 * w
    )

    kernel = np.ones(
        (1, close_width),
        dtype=np.uint8
    )

    column_mask = cv2.morphologyEx(
        column_mask.reshape(1, -1),
        cv2.MORPH_CLOSE,
        kernel
    ).ravel()

    regions = []
    start = None

    for x in range(w):
        active = column_mask[x] > 0

        if active and start is None:
            start = x

        if (
            start is not None
            and (
                not active
                or x == w - 1
            )
        ):
            end = x if active else x - 1
            region_width = end - start + 1

            if region_width > 0.04 * w:
                regions.append(
                    (start, end)
                )

            start = None

    left_candidates = []
    right_candidates = []

    for region in regions:
        centre = (
            region[0] + region[1]
        ) / 2.0

        if centre < 0.45 * w:
            left_candidates.append(
                region
            )
        elif centre > 0.55 * w:
            right_candidates.append(
                region
            )

    if not left_candidates:
        raise RuntimeError(
            "Could not detect left ruler."
        )

    if not right_candidates:
        raise RuntimeError(
            "Could not detect right ruler."
        )

    left_region = max(
        left_candidates,
        key=lambda r: r[1] - r[0]
    )

    right_region = max(
        right_candidates,
        key=lambda r: r[1] - r[0]
    )

    def make_box(region):
        xs, xe = region

        roi = gray[:, xs:xe + 1]

        row_dark_fraction = np.mean(
            roi < RULER_DARK_THRESHOLD,
            axis=1
        )

        ys = np.where(
            row_dark_fraction > 0.30
        )[0]

        if len(ys) == 0:
            raise RuntimeError(
                "Could not determine ruler vertical extent."
            )

        top = int(ys.min())
        bottom = int(ys.max())

        width = xe - xs + 1
        height = bottom - top + 1

        return (
            int(xs),
            int(top),
            int(width),
            int(height),
            int(width * height),
        )

    return (
        make_box(left_region),
        make_box(right_region),
    )


# ============================================================
# MAJOR RULER MARKS
# ============================================================

def estimate_major_spacing(
    marks,
    ruler_box
):
    x, y, w, h, _ = ruler_box
    marks = sorted(marks)

    if len(marks) < 2:
        raise RuntimeError(
            "Not enough major marks."
        )

    upper_limit = y + 0.82 * h

    upper_marks = [
        value
        for value in marks
        if value <= upper_limit
    ]

    if len(upper_marks) < 3:
        upper_marks = marks[
            :max(
                3,
                int(0.80 * len(marks))
            )
        ]

    differences = np.diff(
        upper_marks
    ).astype(np.float32)

    plausible = differences[
        (
            differences > 0.02 * h
        )
        &
        (
            differences < 0.10 * h
        )
    ]

    if len(plausible) > 0:
        return float(
            np.median(plausible)
        )

    return float(
        np.median(differences)
    )


def regularize_major_marks(
    marks,
    ruler_box
):
    marks = sorted(marks)

    if len(marks) < 3:
        return marks

    spacing = estimate_major_spacing(
        marks,
        ruler_box
    )

    cleaned = [marks[0]]

    for current in marks[1:]:
        previous = cleaned[-1]
        gap = current - previous

        if gap < 0.55 * spacing:
            continue

        if gap < 1.55 * spacing:
            cleaned.append(current)
            continue

        intervals = max(
            1,
            int(round(gap / spacing))
        )

        if intervals >= 2:
            for k in range(
                1,
                intervals
            ):
                reconstructed = int(
                    round(
                        previous
                        + gap * k / intervals
                    )
                )
                cleaned.append(
                    reconstructed
                )

        cleaned.append(current)

    return cleaned


def detect_major_ruler_marks(
    image,
    ruler_box,
    side
):
    x, y, w, h, _ = ruler_box

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    roi = gray[
        y:y + h,
        x:x + w
    ]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(4, 16)
    )

    enhanced = clahe.apply(roi)

    block_size = odd_number(
        0.018 * h,
        minimum=15
    )

    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        -3,
    )

    if side == "left":
        band_x0 = int(0.58 * w)
        band_x1 = int(0.82 * w)

    elif side == "right":
        band_x0 = int(0.18 * w)
        band_x1 = int(0.42 * w)

    else:
        raise ValueError(
            "side must be left or right"
        )

    band = binary[
        :,
        band_x0:band_x1
    ]

    kernel_width = max(
        5,
        int(0.05 * w)
    )

    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_width, 1)
        )
    )

    horizontal = cv2.morphologyEx(
        band,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    row_strength = np.sum(
        horizontal > 0,
        axis=1
    ).astype(np.float32)

    row_strength = smooth_1d(
        row_strength,
        odd_number(0.004 * h)
    )

    margin = max(
        5,
        int(0.005 * h)
    )

    row_strength[:margin] = 0
    row_strength[-margin:] = 0

    positive = row_strength[
        row_strength > 0
    ]

    if len(positive) == 0:
        return []

    threshold = max(
        np.percentile(
            positive,
            60
        ),
        0.18 * (
            band_x1 - band_x0
        )
    )

    min_distance = max(
        10,
        int(
            MAJOR_MIN_DISTANCE_REL * h
        )
    )

    peaks = find_peaks_1d(
        row_strength,
        min_distance,
        threshold
    )

    marks = [
        int(y + p)
        for p in peaks
    ]

    return regularize_major_marks(
        marks,
        ruler_box
    )


def pair_major_marks(
    left_marks,
    right_marks,
    image_height
):
    tolerance = max(
        5,
        int(
            MAJOR_PAIR_TOLERANCE_REL
            * image_height
        )
    )

    paired = []
    used_right = set()

    for y_left in left_marks:
        candidates = []

        for j, y_right in enumerate(
            right_marks
        ):
            if j in used_right:
                continue

            candidates.append(
                (
                    abs(y_left - y_right),
                    j,
                    y_right,
                )
            )

        if not candidates:
            continue

        difference, j, y_right = min(
            candidates,
            key=lambda item: item[0]
        )

        if difference <= tolerance:
            paired.append(
                (
                    int(y_left),
                    int(y_right),
                )
            )
            used_right.add(j)

    paired.sort(
        key=lambda pair:
        (
            pair[0] + pair[1]
        ) / 2.0
    )

    return paired


def select_top_down_major_scale(
    major_pairs
):
    expected_values = list(
        range(
            TOP_MAJOR_VALUE,
            BOTTOM_MAJOR_VALUE - 1,
            -MAJOR_STEP,
        )
    )

    expected_count = len(
        expected_values
    )

    if len(major_pairs) < expected_count:
        raise RuntimeError(
            "Not enough paired major lines. "
            f"Found {len(major_pairs)}, "
            f"required {expected_count}."
        )

    best_start = None
    best_score = None

    for start in range(
        len(major_pairs)
        - expected_count
        + 1
    ):
        window = major_pairs[
            start:
            start + expected_count
        ]

        left_values = np.asarray(
            [
                pair[0]
                for pair in window
            ],
            dtype=np.float32,
        )

        right_values = np.asarray(
            [
                pair[1]
                for pair in window
            ],
            dtype=np.float32,
        )

        left_steps = np.diff(
            left_values
        )
        right_steps = np.diff(
            right_values
        )

        if (
            np.any(left_steps <= 0)
            or np.any(right_steps <= 0)
        ):
            continue

        median_left = float(
            np.median(left_steps)
        )

        median_right = float(
            np.median(right_steps)
        )

        left_error = np.mean(
            np.abs(
                left_steps - median_left
            )
            /
            max(median_left, 1.0)
        )

        right_error = np.mean(
            np.abs(
                right_steps - median_right
            )
            /
            max(median_right, 1.0)
        )

        pair_error = np.mean(
            np.abs(
                (
                    left_steps
                    / max(median_left, 1.0)
                )
                -
                (
                    right_steps
                    / max(median_right, 1.0)
                )
            )
        )

        score = (
            left_error
            + right_error
            + 0.5 * pair_error
        )

        if (
            best_score is None
            or score < best_score
        ):
            best_score = score
            best_start = start

    if best_start is None:
        raise RuntimeError(
            "Could not find regular 75-to-5 sequence."
        )

    selected = major_pairs[
        best_start:
        best_start + expected_count
    ]

    major_scale = []

    for value, pair in zip(
        expected_values,
        selected
    ):
        major_scale.append(
            {
                "value": int(value),
                "left_y": float(pair[0]),
                "right_y": float(pair[1]),
            }
        )

    return major_scale


def calculate_zero_anchor(
    major_scale
):
    interval_count = min(
        BOTTOM_REFERENCE_INTERVALS,
        len(major_scale) - 1
    )

    recent = major_scale[
        -(interval_count + 1):
    ]

    left_steps = []
    right_steps = []

    for i in range(
        len(recent) - 1
    ):
        left_steps.append(
            recent[i + 1]["left_y"]
            - recent[i]["left_y"]
        )

        right_steps.append(
            recent[i + 1]["right_y"]
            - recent[i]["right_y"]
        )

    left_spacing = float(
        np.median(left_steps)
    )

    right_spacing = float(
        np.median(right_steps)
    )

    five = major_scale[-1]

    return {
        "value": 0,
        "left_y":
            five["left_y"]
            + left_spacing,
        "right_y":
            five["right_y"]
            + right_spacing,
    }


def calculate_top_minor_lines(
    major_scale
):
    interval_count = min(
        TOP_MINOR_REFERENCE_INTERVALS,
        len(major_scale) - 1
    )

    early = major_scale[
        :interval_count + 1
    ]

    left_steps = []
    right_steps = []

    for i in range(
        len(early) - 1
    ):
        left_steps.append(
            early[i + 1]["left_y"]
            - early[i]["left_y"]
        )

        right_steps.append(
            early[i + 1]["right_y"]
            - early[i]["right_y"]
        )

    left_major_spacing = float(
        np.median(left_steps)
    )

    right_major_spacing = float(
        np.median(right_steps)
    )

    left_unit = (
        left_major_spacing
        / MAJOR_STEP
    )

    right_unit = (
        right_major_spacing
        / MAJOR_STEP
    )

    line_75 = major_scale[0]
    result = []

    for value in range(
        TOP_MAJOR_VALUE + 1,
        TOP_VISIBLE_VALUE + 1,
    ):
        delta = (
            value - TOP_MAJOR_VALUE
        )

        result.append(
            {
                "value": int(value),
                "left_y":
                    line_75["left_y"]
                    - delta * left_unit,
                "right_y":
                    line_75["right_y"]
                    - delta * right_unit,
            }
        )

    return result


def generate_measurement_lines(
    major_scale
):
    zero = calculate_zero_anchor(
        major_scale
    )

    anchors = (
        list(major_scale)
        + [zero]
    )

    lines = []

    for i in range(
        len(anchors) - 1
    ):
        top = anchors[i]
        bottom = anchors[i + 1]

        top_value = top["value"]
        bottom_value = bottom["value"]

        difference = (
            top_value - bottom_value
        )

        for j in range(difference):
            alpha = j / difference
            value = top_value - j

            left_y = (
                top["left_y"]
                + alpha
                * (
                    bottom["left_y"]
                    - top["left_y"]
                )
            )

            right_y = (
                top["right_y"]
                + alpha
                * (
                    bottom["right_y"]
                    - top["right_y"]
                )
            )

            lines.append(
                {
                    "value": int(value),
                    "left_y": float(left_y),
                    "right_y": float(right_y),
                }
            )

    lines.append(
        {
            "value": 0,
            "left_y": float(
                zero["left_y"]
            ),
            "right_y": float(
                zero["right_y"]
            ),
        }
    )

    lines.extend(
        calculate_top_minor_lines(
            major_scale
        )
    )

    lines.sort(
        key=lambda item:
        item["value"],
        reverse=True
    )

    return lines


# ============================================================
# SAMPLE DETECTION
# ============================================================

def detect_coarse_sample_edges(
    image
):
    h, w = image.shape[:2]

    bgr = image.astype(
        np.float32
    )

    B = bgr[:, :, 0]
    G = bgr[:, :, 1]
    R = bgr[:, :, 2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    search_x0 = int(
        SAMPLE_SEARCH_LEFT * w
    )

    search_x1 = int(
        SAMPLE_SEARCH_RIGHT * w
    )

    red_score = (
        (R - G)
        + 0.20 * (R - B)
        - 0.20 * gray
    )

    roi_score = red_score[
        :,
        search_x0:search_x1
    ]

    score_u8 = cv2.normalize(
        roi_score,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, mask = cv2.threshold(
        score_u8,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    k = odd_number(
        0.004 * w
    )

    kernel = np.ones(
        (k, k),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    (
        number_labels,
        labels,
        stats,
        centroids,
    ) = cv2.connectedComponentsWithStats(
        mask
    )

    target_x = (
        search_x1 - search_x0
    ) / 2.0

    best_label = None
    best_value = None

    for label in range(
        1,
        number_labels
    ):
        x, y, bw, bh, area = (
            stats[label]
        )
        cx, cy = centroids[label]

        if bh < 0.35 * h:
            continue

        if area < 0.0005 * h * w:
            continue

        centre_error = abs(
            cx - target_x
        )

        value = (
            centre_error
            - 0.02 * bh
        )

        if (
            best_value is None
            or value < best_value
        ):
            best_value = value
            best_label = label

    if best_label is None:
        raise RuntimeError(
            "Could not locate porous sample."
        )

    sample_mask = (
        labels == best_label
    )

    ys = []
    left_xs = []
    right_xs = []

    for yy in range(h):
        xs_local = np.where(
            sample_mask[yy]
        )[0]

        if len(xs_local) < 4:
            continue

        ys.append(yy)

        left_xs.append(
            search_x0
            + np.percentile(
                xs_local,
                5
            )
        )

        right_xs.append(
            search_x0
            + np.percentile(
                xs_local,
                95
            )
        )

    ys = np.asarray(
        ys,
        dtype=np.float32
    )

    left_xs = np.asarray(
        left_xs,
        dtype=np.float32
    )

    right_xs = np.asarray(
        right_xs,
        dtype=np.float32
    )

    if len(ys) < 0.05 * h:
        raise RuntimeError(
            "Not enough points for coarse sample detection."
        )

    a_left, b_left = np.polyfit(
        ys,
        left_xs,
        1
    )

    a_right, b_right = np.polyfit(
        ys,
        right_xs,
        1
    )

    return (
        (
            float(a_left),
            float(b_left),
        ),
        (
            float(a_right),
            float(b_right),
        ),
        int(
            np.percentile(
                ys,
                2
            )
        ),
        int(
            np.percentile(
                ys,
                98
            )
        ),
    )


def robust_fit_x_as_function_of_y(
    xs,
    ys
):
    xs = np.asarray(
        xs,
        dtype=np.float32
    )

    ys = np.asarray(
        ys,
        dtype=np.float32
    )

    points = np.column_stack(
        (xs, ys)
    ).astype(np.float32)

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_HUBER,
        0,
        0.01,
        0.01,
    ).flatten()

    if abs(vy) < 1e-8:
        raise RuntimeError(
            "Invalid sample edge fit."
        )

    a = vx / vy
    b = x0 - a * y0

    predicted = (
        a * ys + b
    )

    residuals = np.abs(
        xs - predicted
    )

    median_residual = np.median(
        residuals
    )

    mad = np.median(
        np.abs(
            residuals
            - median_residual
        )
    ) + 1e-6

    threshold = max(
        3.0,
        3.0 * 1.4826 * mad
    )

    keep = (
        residuals <= threshold
    )

    if np.sum(keep) >= 20:
        points = np.column_stack(
            (
                xs[keep],
                ys[keep],
            )
        ).astype(np.float32)

        vx, vy, x0, y0 = cv2.fitLine(
            points,
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        ).flatten()

        if abs(vy) > 1e-8:
            a = vx / vy
            b = x0 - a * y0

    return (
        float(a),
        float(b),
    )


def refine_sample_edges(
    image,
    coarse_left,
    coarse_right,
    sample_y_min,
    sample_y_max
):
    h, w = image.shape[:2]

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    L = lab[:, :, 0]
    A = lab[:, :, 1]

    score = (
        A - 0.12 * L
    )

    blur_x = odd_number(
        SAMPLE_EDGE_BLUR_X_REL * w
    )

    blur_y = odd_number(
        SAMPLE_EDGE_BLUR_Y_REL * h
    )

    score = cv2.GaussianBlur(
        score,
        (blur_x, blur_y),
        0
    )

    gradient_x = cv2.Sobel(
        score,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    row_step = max(
        1,
        int(
            SAMPLE_EDGE_ROW_STEP_REL * h
        )
    )

    detected_y = []
    detected_left = []
    detected_right = []
    detected_strength = []

    for y in range(
        max(sample_y_min, 5),
        min(sample_y_max, h - 5),
        row_step,
    ):
        coarse_x_left = (
            coarse_left[0] * y
            + coarse_left[1]
        )

        coarse_x_right = (
            coarse_right[0] * y
            + coarse_right[1]
        )

        coarse_width = (
            coarse_x_right
            - coarse_x_left
        )

        if coarse_width <= 0:
            continue

        centre_x = (
            coarse_x_left
            + coarse_x_right
        ) / 2.0

        left_start = int(
            round(
                centre_x
                - 0.55 * coarse_width
            )
        )

        left_end = int(
            round(
                centre_x
                - 0.02 * coarse_width
            )
        )

        right_start = int(
            round(
                centre_x
                + 0.02 * coarse_width
            )
        )

        right_end = int(
            round(
                centre_x
                + 0.55 * coarse_width
            )
        )

        left_start = max(
            1,
            left_start
        )

        left_end = min(
            w - 2,
            left_end
        )

        right_start = max(
            1,
            right_start
        )

        right_end = min(
            w - 2,
            right_end
        )

        if left_end <= left_start:
            continue

        if right_end <= right_start:
            continue

        left_gradient = gradient_x[
            y,
            left_start:left_end + 1
        ]

        right_gradient = gradient_x[
            y,
            right_start:right_end + 1
        ]

        left_edge = (
            left_start
            + int(
                np.argmax(
                    left_gradient
                )
            )
        )

        right_edge = (
            right_start
            + int(
                np.argmin(
                    right_gradient
                )
            )
        )

        detected_width = (
            right_edge - left_edge
        )

        minimum_width = (
            SAMPLE_WIDTH_MIN_RATIO
            * coarse_width
        )

        maximum_width = (
            SAMPLE_WIDTH_MAX_RATIO
            * coarse_width
        )

        if detected_width < minimum_width:
            continue

        if detected_width > maximum_width:
            continue

        left_strength = float(
            gradient_x[
                y,
                left_edge
            ]
        )

        right_strength = float(
            -gradient_x[
                y,
                right_edge
            ]
        )

        if (
            left_strength <= 0
            or right_strength <= 0
        ):
            continue

        detected_y.append(y)
        detected_left.append(left_edge)
        detected_right.append(right_edge)

        detected_strength.append(
            min(
                left_strength,
                right_strength
            )
        )

    minimum_required = max(
        30,
        int(0.03 * h)
    )

    if len(detected_y) < minimum_required:
        raise RuntimeError(
            "Not enough points to refine sample edges. "
            f"Found {len(detected_y)}, "
            f"required {minimum_required}."
        )

    detected_y = np.asarray(
        detected_y,
        dtype=np.float32
    )

    detected_left = np.asarray(
        detected_left,
        dtype=np.float32
    )

    detected_right = np.asarray(
        detected_right,
        dtype=np.float32
    )

    detected_strength = np.asarray(
        detected_strength,
        dtype=np.float32
    )

    strength_threshold = np.percentile(
        detected_strength,
        15
    )

    keep = (
        detected_strength
        >= strength_threshold
    )

    detected_y = detected_y[keep]
    detected_left = detected_left[keep]
    detected_right = detected_right[keep]

    widths = (
        detected_right
        - detected_left
    )

    median_width = np.median(
        widths
    )

    width_mad = np.median(
        np.abs(
            widths - median_width
        )
    ) + 1e-6

    width_tolerance = max(
        3.0,
        3.5 * 1.4826 * width_mad
    )

    keep = (
        np.abs(
            widths - median_width
        )
        <= width_tolerance
    )

    detected_y = detected_y[keep]
    detected_left = detected_left[keep]
    detected_right = detected_right[keep]

    if len(detected_y) < 20:
        raise RuntimeError(
            "Too many sample-edge points rejected."
        )

    left_line = (
        robust_fit_x_as_function_of_y(
            detected_left,
            detected_y
        )
    )

    right_line = (
        robust_fit_x_as_function_of_y(
            detected_right,
            detected_y
        )
    )

    return (
        left_line,
        right_line,
        int(
            np.percentile(
                detected_y,
                2
            )
        ),
        int(
            np.percentile(
                detected_y,
                98
            )
        ),
    )


def detect_sample_edges(image):
    (
        coarse_left,
        coarse_right,
        y_min,
        y_max,
    ) = detect_coarse_sample_edges(
        image
    )

    return refine_sample_edges(
        image,
        coarse_left,
        coarse_right,
        y_min,
        y_max,
    )


def x_from_line(line, y):
    a, b = line

    return int(
        round(
            a * y + b
        )
    )


def get_sample_center_line(
    left_line,
    right_line
):
    return (
        (
            left_line[0]
            + right_line[0]
        ) / 2.0,
        (
            left_line[1]
            + right_line[1]
        ) / 2.0,
    )


# ============================================================
# SCALE CALIBRATION AT SAMPLE CENTRE
# ============================================================

def measurement_line_intersection(
    left_y,
    right_y,
    x_left,
    x_right,
    sample_center_line,
):
    dx = x_right - x_left

    if abs(dx) < 1e-9:
        return None

    m = (
        right_y - left_y
    ) / dx

    c = (
        left_y
        - m * x_left
    )

    a, b = sample_center_line

    denominator = (
        1.0 - m * a
    )

    if abs(denominator) < 1e-9:
        return None

    y_cross = (
        m * b + c
    ) / denominator

    x_cross = (
        a * y_cross + b
    )

    return (
        float(x_cross),
        float(y_cross),
    )


def build_scale_positions(
    measurement_lines,
    left_ruler,
    right_ruler,
    sample_center_line,
):
    lx, ly, lw, lh, _ = left_ruler
    rx, ry, rw, rh, _ = right_ruler

    x_start = lx + lw
    x_end = rx

    positions = []

    for line in measurement_lines:
        result = measurement_line_intersection(
            line["left_y"],
            line["right_y"],
            x_start,
            x_end,
            sample_center_line,
        )

        if result is None:
            continue

        x_cross, y_cross = result

        positions.append(
            {
                "value": int(
                    line["value"]
                ),
                "x": float(x_cross),
                "y": float(y_cross),
                "left_y": float(
                    line["left_y"]
                ),
                "right_y": float(
                    line["right_y"]
                ),
            }
        )

    positions.sort(
        key=lambda item:
        item["value"]
    )

    return positions


def front_y_to_height_mm(
    y_front,
    scale_positions,
):
    if len(scale_positions) < 2:
        return np.nan

    ordered = sorted(
        scale_positions,
        key=lambda item: item["y"]
    )

    ys = np.asarray(
        [
            item["y"]
            for item in ordered
        ],
        dtype=np.float32,
    )

    ruler_values = np.asarray(
        [
            item["value"]
            for item in ordered
        ],
        dtype=np.float32,
    )

    ruler_value = np.interp(
        float(y_front),
        ys,
        ruler_values,
    )

    ruler_value = np.clip(
        ruler_value,
        0.0,
        float(TOP_VISIBLE_VALUE),
    )

    return float(
        ruler_value
        * MM_PER_RULER_UNIT
    )


def height_mm_to_y(
    height_mm,
    scale_positions,
):
    if len(scale_positions) < 2:
        return np.nan

    ordered = sorted(
        scale_positions,
        key=lambda item:
        item["value"]
    )

    heights = np.asarray(
        [
            item["value"]
            * MM_PER_RULER_UNIT
            for item in ordered
        ],
        dtype=np.float64,
    )

    ys = np.asarray(
        [
            item["y"]
            for item in ordered
        ],
        dtype=np.float64,
    )

    height_mm = np.clip(
        float(height_mm),
        heights[0],
        heights[-1],
    )

    return float(
        np.interp(
            height_mm,
            heights,
            ys,
        )
    )


def find_closest_scale_line(
    y_front,
    scale_positions,
):
    if not scale_positions:
        return None

    return min(
        scale_positions,
        key=lambda item:
        abs(
            y_front - item["y"]
        )
    )


# ============================================================
# FRONT DETECTION
# ============================================================

def prepare_front_geometry(
    image_shape,
    left_sample_line,
    right_sample_line,
    sample_y_min,
    sample_y_max,
    zero_y,
):
    h, w = image_shape[:2]

    y_start = max(
        sample_y_min,
        int(
            FRONT_SEARCH_TOP * h
        ),
    )

    # Reflections below the calculated fluid-level zero line are
    # excluded from the analysis completely.
    y_end = min(
        sample_y_max,
        int(
            FRONT_SEARCH_BOTTOM * h
        ),
        int(
            np.floor(zero_y)
        ),
    )

    if y_end <= y_start:
        raise RuntimeError(
            "Invalid front-analysis region."
        )

    x1_values = []
    x2_values = []

    for y in range(
        y_start,
        y_end + 1
    ):
        left = (
            left_sample_line[0] * y
            + left_sample_line[1]
        )

        right = (
            right_sample_line[0] * y
            + right_sample_line[1]
        )

        sample_width = (
            right - left
        )

        x1 = int(
            round(
                left
                + SAMPLE_INNER_MARGIN
                * sample_width
            )
        )

        x2 = int(
            round(
                right
                - SAMPLE_INNER_MARGIN
                * sample_width
            )
        )

        x1_values.append(x1)
        x2_values.append(x2)

    crop_x0 = max(
        0,
        min(x1_values) - 3
    )

    crop_x1 = min(
        w,
        max(x2_values) + 4
    )

    crop_width = (
        crop_x1 - crop_x0
    )

    row_count = (
        y_end - y_start + 1
    )

    mask = np.zeros(
        (
            row_count,
            crop_width,
        ),
        dtype=bool,
    )

    for i, (x1, x2) in enumerate(
        zip(
            x1_values,
            x2_values
        )
    ):
        local_x1 = max(
            0,
            x1 - crop_x0
        )

        local_x2 = min(
            crop_width,
            x2 - crop_x0
        )

        if local_x2 > local_x1:
            mask[
                i,
                local_x1:local_x2
            ] = True

    return {
        "y_start": int(y_start),
        "y_end": int(y_end),
        "x_start": int(crop_x0),
        "x_end": int(crop_x1),
        "mask": mask,
    }


def detect_liquid_front(
    frame,
    front_geometry,
):
    y_start = front_geometry[
        "y_start"
    ]

    y_end = front_geometry[
        "y_end"
    ]

    x_start = front_geometry[
        "x_start"
    ]

    x_end = front_geometry[
        "x_end"
    ]

    mask = front_geometry[
        "mask"
    ]

    crop = frame[
        y_start:y_end + 1,
        x_start:x_end,
    ]

    lab = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2LAB
    )

    L = lab[:, :, 0].astype(
        np.float32
    )

    masked = np.where(
        mask,
        L,
        np.nan
    )

    profile = np.nanmedian(
        masked,
        axis=1
    )

    valid = np.isfinite(
        profile
    )

    if np.sum(valid) < 20:
        raise RuntimeError(
            "Not enough valid rows for front detection."
        )

    if not np.all(valid):
        indices = np.where(valid)[0]

        profile = np.interp(
            np.arange(len(profile)),
            indices,
            profile[valid],
        )

    kernel = odd_number(
        FRONT_SMOOTH_REL
        * frame.shape[0]
    )

    maximum_kernel = (
        len(profile)
        if len(profile) % 2 == 1
        else len(profile) - 1
    )

    kernel = min(
        kernel,
        maximum_kernel
    )
    kernel = max(3, kernel)

    profile_smooth = smooth_1d(
        profile,
        kernel
    )

    derivative = np.gradient(
        profile_smooth
    )

    # Wet region below the front is darker.
    edge_strength = -derivative

    local_front = int(
        np.argmax(edge_strength)
    )

    y_front = (
        y_start + local_front
    )

    background = np.median(
        edge_strength
    )

    mad = np.median(
        np.abs(
            edge_strength - background
        )
    ) + 1e-6

    confidence_score = (
        edge_strength[local_front]
        - background
    ) / (
        1.4826 * mad
    )

    return (
        int(y_front),
        float(confidence_score),
    )


# ============================================================
# DRAWING
# ============================================================

def draw_static_geometry(
    image,
    left_ruler,
    right_ruler,
    measurement_lines,
    left_sample_line,
    right_sample_line,
    sample_y_min,
    sample_y_max,
):
    out = image.copy()
    h, w = out.shape[:2]

    lx, ly, lw, lh, _ = left_ruler
    rx, ry, rw, rh, _ = right_ruler

    x_start = lx + lw
    x_end = rx

    pad = max(
        5,
        int(
            RULER_BOX_PAD_REL * w
        )
    )

    cv2.rectangle(
        out,
        (
            max(0, lx - pad),
            ly,
        ),
        (
            lx + lw,
            ly + lh,
        ),
        (255, 255, 0),
        2,
    )

    cv2.rectangle(
        out,
        (rx, ry),
        (
            min(
                w - 1,
                rx + rw + pad
            ),
            ry + rh,
        ),
        (255, 255, 0),
        2,
    )

    for line in measurement_lines:
        value = line["value"]

        y_left = int(
            round(
                line["left_y"]
            )
        )

        y_right = int(
            round(
                line["right_y"]
            )
        )

        colour = (255, 0, 0)
        thickness = 1

        if (
            value <= TOP_MAJOR_VALUE
            and value >= BOTTOM_MAJOR_VALUE
            and value % 5 == 0
        ):
            colour = (255, 100, 0)
            thickness = 2

        if value == 0:
            colour = (0, 255, 255)
            thickness = 3

        cv2.line(
            out,
            (x_start, y_left),
            (x_end, y_right),
            colour,
            thickness,
            cv2.LINE_AA,
        )

    y0 = max(
        0,
        sample_y_min
    )

    y1 = min(
        h - 1,
        sample_y_max
    )

    cv2.line(
        out,
        (
            x_from_line(
                left_sample_line,
                y0
            ),
            y0,
        ),
        (
            x_from_line(
                left_sample_line,
                y1
            ),
            y1,
        ),
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.line(
        out,
        (
            x_from_line(
                right_sample_line,
                y0
            ),
            y0,
        ),
        (
            x_from_line(
                right_sample_line,
                y1
            ),
            y1,
        ),
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return out


def draw_front_overlay(
    image,
    left_sample_line,
    right_sample_line,
    y_front,
    height_mm,
    nearest_line,
    confidence_score,
    time_s,
):
    out = image.copy()
    h, w = out.shape[:2]

    confidence_percent = (
        confidence_score_to_percent(
            confidence_score
        )
    )

    x_left = x_from_line(
        left_sample_line,
        y_front
    )

    x_right = x_from_line(
        right_sample_line,
        y_front
    )

    cv2.line(
        out,
        (x_left, y_front),
        (x_right, y_front),
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    if nearest_line is not None:
        cv2.circle(
            out,
            (
                int(
                    round(
                        nearest_line["x"]
                    )
                ),
                int(
                    round(
                        nearest_line["y"]
                    )
                ),
            ),
            6,
            (0, 255, 255),
            -1,
        )

    font_scale = max(
        0.65,
        w / 1300.0
    )

    thickness = max(
        2,
        int(w / 900)
    )

    x_text = int(
        0.02 * w
    )

    y_text = int(
        0.04 * h
    )

    step = int(
        0.035 * h
    )

    cv2.putText(
        out,
        f"time = {time_s:.2f} s",
        (x_text, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        (
            f"tracked height = "
            f"{height_mm:.2f} mm"
        ),
        (
            x_text,
            y_text + step,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    if nearest_line is not None:
        cv2.putText(
            out,
            (
                f"nearest = "
                f"{nearest_line['value']} mm"
            ),
            (
                x_text,
                y_text + 2 * step,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.8,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        (
            f"detection confidence = "
            f"{confidence_percent:.1f}%"
        ),
        (
            x_text,
            y_text + 3 * step,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.75,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA,
    )

    return out


# ============================================================
# CSV / PLOTS
# ============================================================

def save_video_csv(
    output_path,
    results,
):
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "frame",
                "time_s",
                "front_y_raw_px",
                "front_y_smoothed_px",
                "front_y_tracked_px",
                "height_raw_mm",
                "height_smoothed_mm",
                "height_tracked_mm",
                "tracking_anchor",
                "nearest_ruler_mm",
                "confidence_score_raw",
                "detection_confidence_percent",
            ]
        )

        for row in results:
            writer.writerow(
                [
                    row["frame"],
                    f"{row['time_s']:.6f}",
                    f"{row['front_y_raw']:.3f}",
                    f"{row['front_y_smoothed']:.3f}",
                    f"{row['front_y_tracked']:.3f}",
                    f"{row['height_raw_mm']:.6f}",
                    f"{row['height_smoothed_mm']:.6f}",
                    f"{row['height_tracked_mm']:.6f}",
                    int(
                        row["tracking_anchor"]
                    ),
                    f"{row['nearest_ruler_mm']:.3f}",
                    f"{row['confidence_score']:.6f}",
                    f"{row['confidence_percent']:.3f}",
                ]
            )


def save_individual_plot(
    output_path,
    folder_name,
    video_name,
    results,
):
    times = np.asarray(
        [
            row["time_s"]
            for row in results
        ]
    )

    raw = np.asarray(
        [
            row["height_raw_mm"]
            for row in results
        ]
    )

    smoothed = np.asarray(
        [
            row["height_smoothed_mm"]
            for row in results
        ]
    )

    tracked = np.asarray(
        [
            row["height_tracked_mm"]
            for row in results
        ]
    )

    anchor_mask = np.asarray(
        [
            row["tracking_anchor"]
            for row in results
        ],
        dtype=bool,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        times,
        raw,
        linestyle="-",
        linewidth=0.8,
        alpha=0.45,
        label="Raw",
    )

    ax.plot(
        times,
        smoothed,
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label="Centred median",
    )

    ax.plot(
        times,
        tracked,
        linestyle="-",
        linewidth=2.2,
        label="Tracked (3 of 5 confirmed)",
    )

    ax.scatter(
        times[anchor_mask],
        tracked[anchor_mask],
        s=18,
        label="Confirmed tracking anchors",
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Capillary rise [mm]"
    )

    ax.set_title(
        f"{folder_name} | {video_name}"
    )

    ax.grid(
        True,
        alpha=0.30
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI
    )

    plt.close(fig)


def calculate_folder_average(
    video_results
):
    if len(video_results) == 0:
        return None

    if len(video_results) == 1:
        run = video_results[0]

        return {
            "time_s":
                run["time_s"].copy(),
            "mean_mm":
                run["height_tracked_mm"].copy(),
            "std_mm":
                np.zeros_like(
                    run["height_tracked_mm"]
                ),
            "n": 1,
        }

    dt_values = []

    for run in video_results:
        times = run["time_s"]

        if len(times) >= 2:
            dt_values.append(
                np.median(
                    np.diff(times)
                )
            )

    if not dt_values:
        raise RuntimeError(
            "Cannot determine average time step."
        )

    dt = float(
        np.max(dt_values)
    )

    if AVERAGE_USE_COMMON_TIME_ONLY:
        end_time = min(
            float(
                run["time_s"][-1]
            )
            for run in video_results
        )
    else:
        end_time = max(
            float(
                run["time_s"][-1]
            )
            for run in video_results
        )

    common_time = np.arange(
        0.0,
        end_time + 0.5 * dt,
        dt,
    )

    interpolated = []

    for run in video_results:
        # IMPORTANT: average from tracked data.
        curve = np.interp(
            common_time,
            run["time_s"],
            run["height_tracked_mm"],
        )

        interpolated.append(curve)

    matrix = np.vstack(
        interpolated
    )

    return {
        "time_s": common_time,
        "mean_mm":
            np.mean(
                matrix,
                axis=0
            ),
        "std_mm":
            np.std(
                matrix,
                axis=0,
                ddof=0
            ),
        "n": len(video_results),
    }


def save_folder_average_csv(
    output_path,
    average,
):
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "time_s",
                "mean_tracked_height_mm",
                "std_tracked_height_mm",
                "number_of_videos",
            ]
        )

        for time_s, mean, std in zip(
            average["time_s"],
            average["mean_mm"],
            average["std_mm"],
        ):
            writer.writerow(
                [
                    f"{time_s:.6f}",
                    f"{mean:.6f}",
                    f"{std:.6f}",
                    average["n"],
                ]
            )


def save_folder_average_plot(
    output_path,
    folder_name,
    average,
):
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    time = average["time_s"]
    mean = average["mean_mm"]
    std = average["std_mm"]

    mean_line, = ax.plot(
        time,
        mean,
        linestyle="-",
        linewidth=2.2,
        label="Average tracked rise",
    )

    if average["n"] > 1:
        ax.fill_between(
            time,
            mean - std,
            mean + std,
            alpha=0.15,
            label="±1 SD",
        )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Capillary rise [mm]"
    )

    ax.set_title(
        f"{folder_name} | Average"
    )

    ax.grid(
        True,
        alpha=0.30
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI
    )

    plt.close(fig)


def save_folder_comparison_plot(
    output_path,
    folder_name,
    video_results,
    average,
):
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for run in video_results:
        ax.plot(
            run["time_s"],
            run["height_tracked_mm"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label=run["video_name"],
        )

    ax.plot(
        average["time_s"],
        average["mean_mm"],
        linestyle="-",
        linewidth=2.8,
        label=(
            f"Average "
            f"(n={average['n']})"
        ),
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Capillary rise [mm]"
    )

    ax.set_title(
        f"{folder_name} | Comparison"
    )

    ax.grid(
        True,
        alpha=0.30
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI
    )

    plt.close(fig)


def save_all_folder_averages_plot(
    output_path,
    root_name,
    folder_averages,
):
    if not folder_averages:
        return

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for (
        folder_name,
        average
    ) in folder_averages.items():
        ax.plot(
            average["time_s"],
            average["mean_mm"],
            linewidth=2.0,
            label=folder_name,
        )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Average capillary rise [mm]"
    )

    ax.set_title(
        f"{root_name} | Folder averages"
    )

    ax.grid(
        True,
        alpha=0.30
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI
    )

    plt.close(fig)


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    video_path,
    output_folder,
    folder_name,
    log_callback=None,
    progress_callback=None,
):
    video_path = Path(video_path)
    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = video_path.stem
    video_name = video_path.name

    output_video = (
        output_folder
        / f"{stem}_marked.mp4"
    )

    output_csv = (
        output_folder
        / f"{stem}_data.csv"
    )

    output_plot = (
        output_folder
        / f"{stem}_height.png"
    )

    output_geometry = (
        output_folder
        / f"{stem}_geometry.png"
    )

    emit_log(
        log_callback,
        f"\n{folder_name} | {video_name}"
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open: {video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if fps <= 0:
        raise RuntimeError(
            "Invalid video FPS."
        )

    emit_log(
        log_callback,
        (
            f"Resolution: {width} x {height} | "
            f"FPS: {fps:.3f} | "
            f"Frames: {total_frames} | "
            f"Duration: {total_frames / fps:.2f} s"
        )
    )

    success, first_frame = cap.read()

    if not success:
        raise RuntimeError(
            "Could not read first frame."
        )

    emit_log(
        log_callback,
        "Detecting static geometry..."
    )

    (
        left_ruler,
        right_ruler,
    ) = detect_rulers(
        first_frame
    )

    left_major_marks = (
        detect_major_ruler_marks(
            first_frame,
            left_ruler,
            "left"
        )
    )

    right_major_marks = (
        detect_major_ruler_marks(
            first_frame,
            right_ruler,
            "right"
        )
    )

    raw_major_pairs = pair_major_marks(
        left_major_marks,
        right_major_marks,
        height,
    )

    major_scale = (
        select_top_down_major_scale(
            raw_major_pairs
        )
    )

    measurement_lines = (
        generate_measurement_lines(
            major_scale
        )
    )

    (
        left_sample_line,
        right_sample_line,
        sample_y_min,
        sample_y_max,
    ) = detect_sample_edges(
        first_frame
    )

    sample_center_line = (
        get_sample_center_line(
            left_sample_line,
            right_sample_line
        )
    )

    scale_positions = (
        build_scale_positions(
            measurement_lines,
            left_ruler,
            right_ruler,
            sample_center_line,
        )
    )

    zero_items = [
        item
        for item in scale_positions
        if item["value"] == 0
    ]

    if len(zero_items) != 1:
        raise RuntimeError(
            "Could not determine calculated zero."
        )

    zero_y = zero_items[0]["y"]

    front_geometry = (
        prepare_front_geometry(
            first_frame.shape,
            left_sample_line,
            right_sample_line,
            sample_y_min,
            sample_y_max,
            zero_y,
        )
    )

    geometry_overlay = (
        draw_static_geometry(
            first_frame,
            left_ruler,
            right_ruler,
            measurement_lines,
            left_sample_line,
            right_sample_line,
            sample_y_min,
            sample_y_max,
        )
    )

    geometry_check = np.hstack(
        (
            first_frame,
            geometry_overlay,
        )
    )

    cv2.imwrite(
        str(output_geometry),
        geometry_check
    )

    # --------------------------------------------------------
    # PASS 1: raw front measurements
    # --------------------------------------------------------

    emit_log(
        log_callback,
        "Pass 1/2: measuring raw liquid front..."
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    raw_y = []
    confidence_scores = []
    frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        (
            y_front,
            confidence_score,
        ) = detect_liquid_front(
            frame,
            front_geometry
        )

        raw_y.append(
            float(y_front)
        )

        confidence_scores.append(
            float(confidence_score)
        )

        frame_number += 1

        if (
            frame_number % 30 == 0
            or frame_number == total_frames
        ):
            pass_percent = (
                100.0
                * frame_number
                / max(1, total_frames)
            )

            emit_progress(
                progress_callback,
                stage="measure",
                percent=pass_percent,
                frame=frame_number,
                total_frames=total_frames,
                video_name=video_name,
            )

    raw_y = np.asarray(
        raw_y,
        dtype=np.float64
    )

    confidence_scores = np.asarray(
        confidence_scores,
        dtype=np.float64
    )

    actual_frames = len(raw_y)

    if actual_frames == 0:
        raise RuntimeError(
            "No frames were measured."
        )

    times = (
        np.arange(
            actual_frames,
            dtype=np.float64
        )
        / fps
    )

    height_raw_mm = np.asarray(
        [
            front_y_to_height_mm(
                y,
                scale_positions
            )
            for y in raw_y
        ],
        dtype=np.float64,
    )

    smoothed_y = (
        centered_median_filter(
            raw_y,
            TEMPORAL_WINDOW
        )
    )

    height_smoothed_mm = np.asarray(
        [
            front_y_to_height_mm(
                y,
                scale_positions
            )
            for y in smoothed_y
        ],
        dtype=np.float64,
    )

    confidence_percent = np.asarray(
        [
            confidence_score_to_percent(
                score
            )
            for score in confidence_scores
        ],
        dtype=np.float64,
    )

    (
        height_tracked_mm,
        tracking_anchor_mask,
        confirmation_records,
    ) = build_monotonic_tracked_curve(
        times,
        height_raw_mm,
        confidence_percent,
    )

    tracked_y = np.asarray(
        [
            height_mm_to_y(
                value,
                scale_positions
            )
            for value in height_tracked_mm
        ],
        dtype=np.float64,
    )

    nearest_ruler_mm = []

    for y in tracked_y:
        nearest = (
            find_closest_scale_line(
                y,
                scale_positions
            )
        )

        if nearest is None:
            nearest_ruler_mm.append(
                np.nan
            )
        else:
            nearest_ruler_mm.append(
                float(
                    nearest["value"]
                    * MM_PER_RULER_UNIT
                )
            )

    nearest_ruler_mm = np.asarray(
        nearest_ruler_mm,
        dtype=np.float64,
    )

    emit_log(
        log_callback,
        (
            f"Tracking anchors accepted: "
            f"{int(np.sum(tracking_anchor_mask))} | "
            f"Final tracked height: "
            f"{height_tracked_mm[-1]:.3f} mm"
        )
    )

    results = []

    for i in range(actual_frames):
        results.append(
            {
                "frame": int(i),
                "time_s": float(
                    times[i]
                ),
                "front_y_raw": float(
                    raw_y[i]
                ),
                "front_y_smoothed": float(
                    smoothed_y[i]
                ),
                "front_y_tracked": float(
                    tracked_y[i]
                ),
                "height_raw_mm": float(
                    height_raw_mm[i]
                ),
                "height_smoothed_mm": float(
                    height_smoothed_mm[i]
                ),
                "height_tracked_mm": float(
                    height_tracked_mm[i]
                ),
                "tracking_anchor": bool(
                    tracking_anchor_mask[i]
                ),
                "nearest_ruler_mm": float(
                    nearest_ruler_mm[i]
                ),
                "confidence_score": float(
                    confidence_scores[i]
                ),
                "confidence_percent": float(
                    confidence_percent[i]
                ),
            }
        )

    # --------------------------------------------------------
    # PASS 2: marked video
    # --------------------------------------------------------

    if WRITE_MARKED_VIDEO:
        emit_log(
            log_callback,
            "Pass 2/2: writing marked video..."
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        fourcc = (
            cv2.VideoWriter_fourcc(
                *VIDEO_CODEC
            )
        )

        writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Could not create marked video."
            )

        frame_number = 0

        while frame_number < actual_frames:
            success, frame = cap.read()

            if not success:
                break

            y_display = int(
                round(
                    tracked_y[
                        frame_number
                    ]
                )
            )

            nearest = (
                find_closest_scale_line(
                    tracked_y[
                        frame_number
                    ],
                    scale_positions,
                )
            )

            static_frame = (
                draw_static_geometry(
                    frame,
                    left_ruler,
                    right_ruler,
                    measurement_lines,
                    left_sample_line,
                    right_sample_line,
                    sample_y_min,
                    sample_y_max,
                )
            )

            marked = draw_front_overlay(
                static_frame,
                left_sample_line,
                right_sample_line,
                y_display,
                height_tracked_mm[
                    frame_number
                ],
                nearest,
                confidence_scores[
                    frame_number
                ],
                times[
                    frame_number
                ],
            )

            writer.write(marked)
            frame_number += 1

            if (
                frame_number % 30 == 0
                or frame_number == actual_frames
            ):
                pass_percent = (
                    100.0
                    * frame_number
                    / actual_frames
                )

                emit_progress(
                    progress_callback,
                    stage="write",
                    percent=pass_percent,
                    frame=frame_number,
                    total_frames=actual_frames,
                    video_name=video_name,
                )

        writer.release()

    cap.release()

    save_video_csv(
        output_csv,
        results
    )

    save_individual_plot(
        output_plot,
        folder_name,
        video_name,
        results,
    )

    emit_log(
        log_callback,
        f"Saved outputs for {video_name}"
    )

    return {
        "video_name": video_name,
        "video_stem": stem,
        "time_s": times,
        "height_raw_mm":
            height_raw_mm,
        "height_smoothed_mm":
            height_smoothed_mm,
        # Primary series for all averages:
        "height_tracked_mm":
            height_tracked_mm,
    }


# ============================================================
# BATCH PROCESSING
# ============================================================

def find_experiment_folders(
    root_folder
):
    root_folder = Path(root_folder)

    return sorted(
        [
            folder
            for folder in root_folder.iterdir()
            if (
                folder.is_dir()
                and folder.name != OUTPUT_ROOT_NAME
            )
        ],
        key=lambda p:
        p.name.lower()
    )


def find_videos_in_folder(
    folder
):
    return sorted(
        [
            file
            for file in Path(folder).iterdir()
            if (
                file.is_file()
                and file.suffix.lower() == ".mp4"
            )
        ],
        key=lambda p:
        p.name.lower()
    )


def count_all_videos(
    experiment_folders
):
    return sum(
        len(
            find_videos_in_folder(
                folder
            )
        )
        for folder in experiment_folders
    )


def process_experiment_folder(
    folder_path,
    output_root,
    video_counter,
    total_videos,
    log_callback=None,
    progress_callback=None,
):
    folder_path = Path(
        folder_path
    )

    folder_name = (
        folder_path.name
    )

    output_folder = (
        Path(output_root)
        / folder_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    videos = find_videos_in_folder(
        folder_path
    )

    if not videos:
        emit_log(
            log_callback,
            f"No MP4 files in {folder_name}; skipped."
        )
        return None, video_counter

    emit_log(
        log_callback,
        (
            f"\n=== Folder: {folder_name} | "
            f"{len(videos)} video(s) ==="
        )
    )

    video_results = []

    for video_path in videos:
        video_counter += 1

        emit_progress(
            progress_callback,
            stage="video_start",
            video_index=video_counter,
            total_videos=total_videos,
            folder_name=folder_name,
            video_name=video_path.name,
            percent=0.0,
        )

        try:
            result = process_video(
                video_path,
                output_folder,
                folder_name,
                log_callback=log_callback,
                progress_callback=progress_callback,
            )

            video_results.append(
                result
            )

            emit_progress(
                progress_callback,
                stage="video_done",
                video_index=video_counter,
                total_videos=total_videos,
                folder_name=folder_name,
                video_name=video_path.name,
                percent=100.0,
            )

        except Exception as error:
            emit_log(
                log_callback,
                (
                    f"FAILED: {folder_name} | "
                    f"{video_path.name}\n"
                    f"Reason: {error}"
                )
            )

            emit_progress(
                progress_callback,
                stage="video_failed",
                video_index=video_counter,
                total_videos=total_videos,
                folder_name=folder_name,
                video_name=video_path.name,
                percent=100.0,
            )

    if not video_results:
        emit_log(
            log_callback,
            (
                f"No successful videos in "
                f"{folder_name}."
            )
        )
        return None, video_counter

    average = calculate_folder_average(
        video_results
    )

    average_csv = (
        output_folder
        / f"{folder_name}_average.csv"
    )

    average_plot = (
        output_folder
        / f"{folder_name}_average.png"
    )

    comparison_plot = (
        output_folder
        / f"{folder_name}_comparison.png"
    )

    save_folder_average_csv(
        average_csv,
        average
    )

    save_folder_average_plot(
        average_plot,
        folder_name,
        average,
    )

    save_folder_comparison_plot(
        comparison_plot,
        folder_name,
        video_results,
        average,
    )

    emit_log(
        log_callback,
        (
            f"Saved tracked average and comparison "
            f"for {folder_name}."
        )
    )

    return (
        {
            "folder_name": folder_name,
            "average": average,
            "video_results": video_results,
        },
        video_counter,
    )


def run_batch_analysis(
    root_folder,
    log_callback=None,
    progress_callback=None,
):
    root_folder = Path(
        root_folder
    ).resolve()

    if not root_folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist:\n{root_folder}"
        )

    if not root_folder.is_dir():
        raise RuntimeError(
            f"Selected path is not a folder:\n{root_folder}"
        )

    output_root = (
        root_folder
        / OUTPUT_ROOT_NAME
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    experiment_folders = (
        find_experiment_folders(
            root_folder
        )
    )

    if not experiment_folders:
        raise RuntimeError(
            "No experiment subfolders were found.\n\n"
            "Expected structure:\n"
            "SelectedFolder\\Experiment_A\\video.mp4"
        )

    total_videos = count_all_videos(
        experiment_folders
    )

    if total_videos == 0:
        raise RuntimeError(
            "No MP4 files were found inside the "
            "experiment subfolders."
        )

    emit_log(
        log_callback,
        f"Selected folder: {root_folder}"
    )

    emit_log(
        log_callback,
        (
            f"Experiment folders: "
            f"{len(experiment_folders)}"
        )
    )

    emit_log(
        log_callback,
        f"Total MP4 videos: {total_videos}"
    )

    folder_results = []
    video_counter = 0

    for folder in experiment_folders:
        result, video_counter = (
            process_experiment_folder(
                folder,
                output_root,
                video_counter,
                total_videos,
                log_callback=log_callback,
                progress_callback=progress_callback,
            )
        )

        if result is not None:
            folder_results.append(
                result
            )

    folder_averages = {
        result["folder_name"]:
        result["average"]
        for result in folder_results
    }

    if folder_averages:
        final_plot = (
            output_root
            / "all_folder_averages.png"
        )

        save_all_folder_averages_plot(
            final_plot,
            root_folder.name,
            folder_averages,
        )

        emit_log(
            log_callback,
            (
                "Saved final comparison: "
                f"{final_plot.name}"
            )
        )

    emit_progress(
        progress_callback,
        stage="all_done",
        percent=100.0,
        video_index=total_videos,
        total_videos=total_videos,
    )

    return output_root


# ============================================================
# GUI
# ============================================================

class LiftHeightAnalyzerApp:
    def __init__(self, root):
        self.root = root

        self.root.title(
            "Lift Height Analyzer"
        )

        self.root.geometry(
            "900x650"
        )

        self.root.minsize(
            760,
            520
        )

        self.events = queue.Queue()
        self.worker_thread = None
        self.output_folder = None

        self.folder_var = tk.StringVar()

        if DEFAULT_EXPERIMENTS_FOLDER.exists():
            self.folder_var.set(
                str(
                    DEFAULT_EXPERIMENTS_FOLDER
                )
            )

        self.status_var = tk.StringVar(
            value="Select the folder containing experiment subfolders."
        )

        self.video_var = tk.StringVar(
            value="No analysis running."
        )

        self.build_ui()

        self.root.after(
            100,
            self.poll_events
        )

    def build_ui(self):
        main = ttk.Frame(
            self.root,
            padding=16
        )

        main.pack(
            fill="both",
            expand=True
        )

        title = ttk.Label(
            main,
            text="Lift Height Analyzer",
            font=(
                "Segoe UI",
                18,
                "bold"
            ),
        )

        title.pack(
            anchor="w",
            pady=(0, 8)
        )

        description = ttk.Label(
            main,
            text=(
                "Select the folder that contains your experiment folders. "
                "Each experiment folder should contain one or more MP4 files."
            ),
            wraplength=820,
        )

        description.pack(
            anchor="w",
            pady=(0, 14)
        )

        folder_frame = ttk.LabelFrame(
            main,
            text="Experiments folder",
            padding=10,
        )

        folder_frame.pack(
            fill="x"
        )

        folder_entry = ttk.Entry(
            folder_frame,
            textvariable=self.folder_var,
        )

        folder_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 8)
        )

        self.select_button = ttk.Button(
            folder_frame,
            text="Select folder...",
            command=self.select_folder,
        )

        self.select_button.pack(
            side="right"
        )

        button_frame = ttk.Frame(
            main
        )

        button_frame.pack(
            fill="x",
            pady=(12, 8)
        )

        self.start_button = ttk.Button(
            button_frame,
            text="Start analysis",
            command=self.start_analysis,
        )

        self.start_button.pack(
            side="left"
        )

        self.open_button = ttk.Button(
            button_frame,
            text="Open results folder",
            command=self.open_results,
            state="disabled",
        )

        self.open_button.pack(
            side="left",
            padx=(8, 0)
        )

        progress_frame = ttk.LabelFrame(
            main,
            text="Progress",
            padding=10,
        )

        progress_frame.pack(
            fill="x",
            pady=(6, 8)
        )

        ttk.Label(
            progress_frame,
            textvariable=self.video_var,
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        self.current_progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
        )

        self.current_progress.pack(
            fill="x",
            pady=(0, 8)
        )

        ttk.Label(
            progress_frame,
            text="Overall videos",
        ).pack(
            anchor="w"
        )

        self.overall_progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
        )

        self.overall_progress.pack(
            fill="x",
            pady=(4, 0)
        )

        ttk.Label(
            main,
            textvariable=self.status_var,
        ).pack(
            anchor="w",
            pady=(4, 6)
        )

        log_frame = ttk.LabelFrame(
            main,
            text="Analysis log",
            padding=8,
        )

        log_frame.pack(
            fill="both",
            expand=True
        )

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            height=16,
            state="disabled",
        )

        scrollbar = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_text.yview,
        )

        self.log_text.configure(
            yscrollcommand=scrollbar.set
        )

        self.log_text.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

    def select_folder(self):
        initial = (
            self.folder_var.get()
            if self.folder_var.get()
            else str(APP_FOLDER)
        )

        selected = filedialog.askdirectory(
            parent=self.root,
            title=(
                "Select folder containing experiment subfolders"
            ),
            initialdir=initial,
        )

        if selected:
            self.folder_var.set(
                selected
            )
            self.status_var.set(
                "Ready to analyse."
            )

    def append_log(self, message):
        self.log_text.configure(
            state="normal"
        )

        self.log_text.insert(
            "end",
            str(message) + "\n"
        )

        self.log_text.see(
            "end"
        )

        self.log_text.configure(
            state="disabled"
        )

    def start_analysis(self):
        if (
            self.worker_thread is not None
            and self.worker_thread.is_alive()
        ):
            return

        folder_text = (
            self.folder_var.get().strip()
        )

        if not folder_text:
            messagebox.showerror(
                "No folder selected",
                "Please select the experiments folder first.",
                parent=self.root,
            )
            return

        folder = Path(folder_text)

        if not folder.exists():
            messagebox.showerror(
                "Folder not found",
                (
                    "The selected folder does not exist:\n\n"
                    f"{folder}"
                ),
                parent=self.root,
            )
            return

        self.start_button.configure(
            state="disabled"
        )

        self.select_button.configure(
            state="disabled"
        )

        self.open_button.configure(
            state="disabled"
        )

        self.current_progress["value"] = 0
        self.overall_progress["value"] = 0

        self.status_var.set(
            "Analysis running..."
        )

        self.video_var.set(
            "Preparing analysis..."
        )

        self.log_text.configure(
            state="normal"
        )

        self.log_text.delete(
            "1.0",
            "end"
        )

        self.log_text.configure(
            state="disabled"
        )

        self.output_folder = None

        self.worker_thread = threading.Thread(
            target=self.analysis_worker,
            args=(folder,),
            daemon=True,
        )

        self.worker_thread.start()

    def analysis_worker(self, folder):
        try:
            output = run_batch_analysis(
                folder,
                log_callback=lambda message:
                    self.events.put(
                        (
                            "log",
                            message
                        )
                    ),
                progress_callback=lambda payload:
                    self.events.put(
                        (
                            "progress",
                            payload
                        )
                    ),
            )

            self.events.put(
                (
                    "success",
                    str(output)
                )
            )

        except Exception:
            self.events.put(
                (
                    "error",
                    traceback.format_exc()
                )
            )

    def poll_events(self):
        try:
            while True:
                event_type, payload = (
                    self.events.get_nowait()
                )

                if event_type == "log":
                    self.append_log(
                        payload
                    )

                elif event_type == "progress":
                    self.update_progress(
                        payload
                    )

                elif event_type == "success":
                    self.output_folder = Path(
                        payload
                    )

                    self.start_button.configure(
                        state="normal"
                    )

                    self.select_button.configure(
                        state="normal"
                    )

                    self.open_button.configure(
                        state="normal"
                    )

                    self.current_progress[
                        "value"
                    ] = 100

                    self.overall_progress[
                        "value"
                    ] = 100

                    self.status_var.set(
                        "Analysis finished successfully."
                    )

                    self.video_var.set(
                        "Finished."
                    )

                    self.append_log(
                        (
                            "\nFINISHED\nResults: "
                            f"{self.output_folder}"
                        )
                    )

                    messagebox.showinfo(
                        "Analysis finished",
                        (
                            "Analysis finished successfully.\n\n"
                            f"Results:\n{self.output_folder}"
                        ),
                        parent=self.root,
                    )

                elif event_type == "error":
                    self.start_button.configure(
                        state="normal"
                    )

                    self.select_button.configure(
                        state="normal"
                    )

                    self.status_var.set(
                        "Analysis failed."
                    )

                    self.video_var.set(
                        "Stopped because of an error."
                    )

                    self.append_log(
                        "\nERROR:\n" + payload
                    )

                    messagebox.showerror(
                        "Analysis failed",
                        (
                            "The analysis stopped because of an error.\n\n"
                            "See the analysis log for details."
                        ),
                        parent=self.root,
                    )

        except queue.Empty:
            pass

        self.root.after(
            100,
            self.poll_events
        )

    def update_progress(self, payload):
        stage = payload.get(
            "stage",
            ""
        )

        percent = float(
            payload.get(
                "percent",
                0.0
            )
        )

        video_index = int(
            payload.get(
                "video_index",
                0
            )
        )

        total_videos = int(
            payload.get(
                "total_videos",
                0
            )
        )

        video_name = payload.get(
            "video_name",
            ""
        )

        folder_name = payload.get(
            "folder_name",
            ""
        )

        if stage == "video_start":
            self.current_progress[
                "value"
            ] = 0

            self.video_var.set(
                (
                    f"Video {video_index}/{total_videos}: "
                    f"{folder_name} | {video_name}"
                )
            )

        elif stage == "measure":
            # Measurement occupies first half of current-video
            # progress.
            self.current_progress[
                "value"
            ] = 0.5 * percent

        elif stage == "write":
            # Marked-video writing occupies second half.
            self.current_progress[
                "value"
            ] = 50.0 + 0.5 * percent

        elif stage in (
            "video_done",
            "video_failed",
        ):
            self.current_progress[
                "value"
            ] = 100

            if total_videos > 0:
                self.overall_progress[
                    "value"
                ] = (
                    100.0
                    * video_index
                    / total_videos
                )

        elif stage == "all_done":
            self.current_progress[
                "value"
            ] = 100

            self.overall_progress[
                "value"
            ] = 100

    def open_results(self):
        if (
            self.output_folder is None
            or not self.output_folder.exists()
        ):
            return

        try:
            if os.name == "nt":
                os.startfile(
                    str(
                        self.output_folder
                    )
                )
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(
                    [
                        "open",
                        str(
                            self.output_folder
                        )
                    ]
                )
            else:
                import subprocess
                subprocess.Popen(
                    [
                        "xdg-open",
                        str(
                            self.output_folder
                        )
                    ]
                )

        except Exception as error:
            messagebox.showerror(
                "Could not open folder",
                str(error),
                parent=self.root,
            )


# ============================================================
# START APPLICATION
# ============================================================

def main():
    root = tk.Tk()

    app = LiftHeightAnalyzerApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()
