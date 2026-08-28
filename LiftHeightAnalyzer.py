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

# 1 ruler division = 1 mm
MM_PER_RULER_UNIT = 1.0

TOP_VISIBLE_VALUE = 79

TOP_MAJOR_VALUE = 75

BOTTOM_MAJOR_VALUE = 5

MAJOR_STEP = 5

TOP_MINOR_REFERENCE_INTERVALS = 3

BOTTOM_REFERENCE_INTERVALS = 4


# ============================================================
# DIAGNOSTIC SMOOTHING
# ============================================================

# Used only for the diagnostic "smoothed" curve.
# NOT used for averages.
TEMPORAL_WINDOW = 5


# ============================================================
# MONOTONIC TRACKING
# ============================================================

# Minimum increase over the currently confirmed level before
# a NEW PRIMARY tracking anchor is accepted.
#
# The previous 0.02 mm setting was smaller than only a few image
# pixels and produced hundreds of nearly duplicate anchors.
TRACK_MIN_RISE_MM = 0.10


# Minimum displayed detection confidence required for a frame
# to participate in confirmation.
#
# NOTE:
# This is a heuristic detection-quality percentage,
# not a calibrated probability.
TRACK_MIN_CONFIDENCE_PERCENT = 55.0


# ============================================================
# NEW: 3-OF-5 CONFIRMATION
# ============================================================

# Examine five consecutive frames.
TRACK_CONFIRM_WINDOW = 5

# At least three must support the new level.
TRACK_CONFIRM_REQUIRED = 3

# "Roughly the same":
#
# values within +/- 0.50 mm of a candidate level are treated
# as supporting the same front.
TRACK_CONFIRM_TOLERANCE_MM = 0.50


# ============================================================
# INITIAL NO-RISE / ONSET HANDLING
# ============================================================

# The physical capillary-rise curve starts at zero.
#
# IMPORTANT:
# The raw detector may find a strong static edge very close to
# the liquid level before a real wetting front exists. That is
# useful diagnostic information, but it must NOT become the
# initial tracked capillary height.
TRACK_INITIAL_HEIGHT_MM = 0.0


# Before the first real rise has been confirmed, a persistent
# detected edge below this height is treated as belonging to the
# zero/liquid-level region rather than as a capillary front.
#
# This threshold is used ONLY to establish the FIRST positive
# tracking level. Once rise has started, later tracking is not
# clamped to this value.
# Because the new temporal wetting validation rejects static
# image edges, this can now be much lower than before.
TRACK_ONSET_MIN_HEIGHT_MM = 0.20


# ============================================================
# NO SYNTHETIC MEASUREMENT DATA
# ============================================================
#
# Temporal interpolation/reconstruction is intentionally disabled.
# Missing or unresolved measurement frames remain unresolved.
#
# Spatial interpolation used only for ruler calibration is still
# allowed; it does NOT create time-series measurements.
TRACK_INTERPOLATE_BETWEEN_ANCHORS = False


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

SAMPLE_INNER_MARGIN = 0.20


# ============================================================
# LIQUID FRONT
# ============================================================

FRONT_SMOOTH_REL = 0.035

FRONT_SEARCH_TOP = 0.05

FRONT_SEARCH_BOTTOM = 0.97


# ============================================================
# TEMPORAL WETTING VALIDATION
# ============================================================

# The first frame is treated as the dry/reference appearance.
# Several initial frames are used only to estimate image noise.
BASELINE_NOISE_FRAMES = 5

# Search several spatial edge candidates instead of trusting only
# the strongest edge in a frame.
FRONT_MAX_CANDIDATES = 12

# Minimum spacing between candidate edges, expressed physically.
FRONT_CANDIDATE_MIN_DISTANCE_MM = 0.20

# For every candidate front, compare the change in sample
# brightness immediately ABOVE and BELOW the candidate.
#
# A true wetting front should have:
#   - comparatively dry material above,
#   - darker/wetted material below.
WETTING_COMPARE_WINDOW_MM = 0.80
WETTING_COMPARE_GAP_MM = 0.10

# Minimum temporal darkening of the region below the candidate,
# in OpenCV Lab L-channel units.
WETTING_MIN_DARKENING_L = 1.5

# Minimum wet-vs-dry step in Lab L-channel units.
WETTING_MIN_STEP_L = 1.5

# The required wetting step also scales automatically with the
# measured image noise in the initial frames.
WETTING_NOISE_MULTIPLIER = 4.0

# The spatial edge itself must still be reasonably real.
WETTING_MIN_EDGE_SCORE = 2.0


# ============================================================
# LARGE-JUMP RETROSPECTIVE REVIEW
# ============================================================

# If the normal 3-of-5 tracker suddenly confirms a level much
# higher than the previous level, do not immediately assume that
# everything in between was unobservable.
#
# Instead, inspect the interval again using ONLY front candidates
# that already passed the temporal wetting validation above.
TRACK_REVIEW_TRIGGER_MM = 1.00

# The retrospective review is deliberately weaker than the main
# 3-of-5 rule, but it is applied only to temporally validated
# wetting-front candidates and only inside a gap that ends at a
# later strong 3-of-5 confirmation.
TRACK_REVIEW_WINDOW = 7
TRACK_REVIEW_REQUIRED = 2
TRACK_REVIEW_TOLERANCE_MM = 0.40
TRACK_REVIEW_MIN_RISE_MM = 0.20


# ============================================================
# RETROSPECTIVE ONSET BACKTRACKING
# ============================================================

# Inspect actual earlier frame detections once a later front is
# strongly confirmed. No missing value is calculated or filled.
TRACK_ONSET_BACKTRACK_FRAMES = 24

TRACK_ONSET_BACKTRACK_MIN_CONFIDENCE_PERCENT = 50.0

# Raw onset candidates may have weaker wetting contrast than the
# normal validator, but must still show positive temporal evidence.
TRACK_ONSET_BACKTRACK_MIN_WETTING_SCORE = 0.10

# Maximum allowed measured rise per frame when following an actual
# trajectory backwards from a later confirmed measurement.
TRACK_ONSET_BACKTRACK_MAX_RISE_PER_FRAME_MM = 0.20

# Unresolved frames may occur between actual onset observations.
TRACK_ONSET_BACKTRACK_MAX_CONSECUTIVE_MISSES = 3


# ============================================================
# CONFIDENCE DISPLAY
# ============================================================

CONFIDENCE_PERCENT_SCALE = 5.0


# ============================================================
# PLOTTING
# ============================================================

FIGURE_DPI = 200

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

    value = max(
        minimum,
        int(round(value))
    )

    if value % 2 == 0:
        value += 1

    return value


def smooth_1d(signal, kernel_size):

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    if len(signal) < 3:
        return signal.copy()

    kernel_size = odd_number(
        kernel_size
    )

    if kernel_size >= len(signal):

        kernel_size = (
            len(signal)
            if len(signal) % 2 == 1
            else len(signal) - 1
        )

    kernel_size = max(
        3,
        kernel_size
    )

    result = cv2.GaussianBlur(
        signal.reshape(-1, 1),
        (1, kernel_size),
        0
    )

    return result.ravel()


def find_peaks_1d(
    signal,
    min_distance,
    min_value
):

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    candidates = []

    for i in range(
        1,
        len(signal) - 1
    ):

        if (
            signal[i] >= signal[i - 1]
            and
            signal[i] >= signal[i + 1]
            and
            signal[i] >= min_value
        ):

            candidates.append(
                (
                    i,
                    float(signal[i])
                )
            )

    candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    selected = []

    for index, strength in candidates:

        if all(
            abs(index - previous) >= min_distance
            for previous in selected
        ):

            selected.append(
                index
            )

    selected.sort()

    return selected


# ============================================================
# CENTRED MEDIAN
# ============================================================

def centered_median_filter(
    values,
    window_size
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    output = values.copy()

    if window_size <= 1:
        return output

    if window_size % 2 == 0:
        window_size += 1

    if len(values) < window_size:
        return output

    half = (
        window_size // 2
    )

    for i in range(
        half,
        len(values) - half
    ):

        output[i] = np.median(
            values[
                i - half:
                i + half + 1
            ]
        )

    return output


# ============================================================
# CONFIDENCE SCORE -> DISPLAY %
# ============================================================

def confidence_score_to_percent(score):

    if not np.isfinite(score):
        return 0.0

    score = max(
        0.0,
        float(score)
    )

    percentage = (
        100.0
        * (
            1.0
            - np.exp(
                -score
                / CONFIDENCE_PERCENT_SCALE
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
# NEW:
# FIND A CONFIRMED LEVEL INSIDE ONE 5-FRAME WINDOW
# ============================================================


def find_confirmed_level_in_window(
    heights,
    confidences,
    global_indices,
    current_height
):
    """
    Confirm a new level from at least 3 of 5 actual frame-derived
    measurements.

    The cluster median is used only to identify a coherent group.
    The returned height is the exact observed height in the chosen
    representative frame.
    """

    heights = np.asarray(
        heights,
        dtype=np.float64
    )

    confidences = np.asarray(
        confidences,
        dtype=np.float64
    )

    global_indices = np.asarray(
        global_indices,
        dtype=int
    )

    valid = (
        np.isfinite(
            heights
        )
        &
        np.isfinite(
            confidences
        )
        &
        (
            confidences
            >= TRACK_MIN_CONFIDENCE_PERCENT
        )
    )

    if np.sum(
        valid
    ) < TRACK_CONFIRM_REQUIRED:
        return None

    valid_heights = (
        heights[
            valid
        ]
    )

    valid_indices = (
        global_indices[
            valid
        ]
    )

    possible_clusters = []

    for candidate_height in valid_heights:

        support = (
            np.abs(
                valid_heights
                - candidate_height
            )
            <= TRACK_CONFIRM_TOLERANCE_MM
        )

        support_count = int(
            np.sum(
                support
            )
        )

        if support_count < TRACK_CONFIRM_REQUIRED:
            continue

        cluster_heights = (
            valid_heights[
                support
            ]
        )

        cluster_indices = (
            valid_indices[
                support
            ]
        )

        cluster_median = float(
            np.median(
                cluster_heights
            )
        )

        distances = np.abs(
            cluster_heights
            - cluster_median
        )

        closest = float(
            np.min(
                distances
            )
        )

        positions = np.where(
            distances
            <= closest
            + 1e-9
        )[0]

        best_position = min(
            positions,
            key=lambda pos:
            int(
                cluster_indices[
                    pos
                ]
            )
        )

        anchor_index = int(
            cluster_indices[
                best_position
            ]
        )

        observed_height = float(
            cluster_heights[
                best_position
            ]
        )

        if (
            current_height
            <= TRACK_INITIAL_HEIGHT_MM
            + 1e-9
            and
            observed_height
            < TRACK_ONSET_MIN_HEIGHT_MM
        ):
            continue

        if (
            observed_height
            < current_height
            + TRACK_MIN_RISE_MM
        ):
            continue

        possible_clusters.append(
            {
                "height":
                    observed_height,

                "cluster_median":
                    cluster_median,

                "anchor_index":
                    anchor_index,

                "support_indices":
                    cluster_indices.copy(),

                "support_count":
                    support_count,

                "type":
                    "confirmed"
            }
        )

    if not possible_clusters:
        return None

    return max(
        possible_clusters,
        key=lambda item:
        (
            item[
                "cluster_median"
            ],
            item[
                "support_count"
            ],
            -item[
                "anchor_index"
            ]
        )
    )



def find_reviewed_anchors_between(
    heights,
    confidences,
    start_index,
    start_height,
    end_index,
    end_height
):
    """
    Review a large jump using only actual temporally validated
    frame measurements.

    The 2-of-7 cluster median is used only for grouping.
    Returned heights are exact observations from actual frames.
    """

    heights = np.asarray(
        heights,
        dtype=np.float64
    )

    confidences = np.asarray(
        confidences,
        dtype=np.float64
    )

    start_index = int(
        start_index
    )

    end_index = int(
        end_index
    )

    if (
        end_index
        <= start_index + 1
        or
        end_height
        <= start_height
        + TRACK_REVIEW_TRIGGER_MM
    ):
        return []

    interval_start = (
        start_index + 1
    )

    interval_end = (
        end_index
    )

    required = int(
        TRACK_REVIEW_REQUIRED
    )

    candidates = []

    for window_start in range(
        interval_start,
        interval_end
    ):

        window_end = min(
            interval_end,
            window_start
            + int(
                TRACK_REVIEW_WINDOW
            )
        )

        if (
            window_end
            - window_start
            < required
        ):
            continue

        h_window = heights[
            window_start:
            window_end
        ]

        c_window = confidences[
            window_start:
            window_end
        ]

        global_indices = np.arange(
            window_start,
            window_end,
            dtype=int
        )

        valid = (
            np.isfinite(
                h_window
            )
            &
            np.isfinite(
                c_window
            )
            &
            (
                c_window
                >= TRACK_MIN_CONFIDENCE_PERCENT
            )
            &
            (
                h_window
                > start_height
                + TRACK_REVIEW_MIN_RISE_MM
            )
            &
            (
                h_window
                < end_height
            )
        )

        if np.sum(
            valid
        ) < required:
            continue

        valid_heights = (
            h_window[
                valid
            ]
        )

        valid_indices = (
            global_indices[
                valid
            ]
        )

        for candidate_height in valid_heights:

            support = (
                np.abs(
                    valid_heights
                    - candidate_height
                )
                <= TRACK_REVIEW_TOLERANCE_MM
            )

            support_count = int(
                np.sum(
                    support
                )
            )

            if support_count < required:
                continue

            cluster_heights = (
                valid_heights[
                    support
                ]
            )

            cluster_indices = (
                valid_indices[
                    support
                ]
            )

            cluster_median = float(
                np.median(
                    cluster_heights
                )
            )

            distances = np.abs(
                cluster_heights
                - cluster_median
            )

            closest = float(
                np.min(
                    distances
                )
            )

            positions = np.where(
                distances
                <= closest
                + 1e-9
            )[0]

            best_position = min(
                positions,
                key=lambda pos:
                int(
                    cluster_indices[
                        pos
                    ]
                )
            )

            anchor_index = int(
                cluster_indices[
                    best_position
                ]
            )

            observed_height = float(
                cluster_heights[
                    best_position
                ]
        )

            if (
                observed_height
                <= start_height
                + TRACK_REVIEW_MIN_RISE_MM
                or
                observed_height
                >= end_height
            ):
                continue

            candidates.append(
                {
                    "anchor_index":
                        anchor_index,

                    "height":
                        observed_height,

                    "cluster_median":
                        cluster_median,

                    "support_indices":
                        cluster_indices.copy(),

                    "support_count":
                        support_count,

                    "type":
                        "reviewed"
                }
            )

    if not candidates:
        return []

    candidates.sort(
        key=lambda item:
        (
            item[
                "anchor_index"
            ],
            -item[
                "support_count"
            ]
        )
    )

    reviewed = []

    current_index = int(
        start_index
    )

    current_height = float(
        start_height
    )

    while True:

        eligible = [
            item
            for item
            in candidates

            if (
                item[
                    "anchor_index"
                ]
                > current_index

                and

                item[
                    "height"
                ]
                >= current_height
                + TRACK_REVIEW_MIN_RISE_MM

                and

                item[
                    "height"
                ]
                < end_height
            )
        ]

        if not eligible:
            break

        earliest_index = min(
            item[
                "anchor_index"
            ]
            for item
            in eligible
        )

        same_time = [
            item
            for item
            in eligible
            if item[
                "anchor_index"
            ] == earliest_index
        ]

        best = max(
            same_time,
            key=lambda item:
            (
                item[
                    "support_count"
                ],
                item[
                    "height"
                ]
            )
        )

        reviewed.append(
            best
        )

        current_index = int(
            best[
                "anchor_index"
            ]
        )

        current_height = float(
            best[
                "height"
            ]
        )

    return reviewed



def find_actual_onset_backtrack(
    raw_heights,
    raw_confidence_percent,
    raw_wetting_scores,
    first_confirmed_index,
    first_confirmed_height
):
    """
    Recover earlier ACTUAL raw frame detections if they form a
    physically continuous trajectory into the first accepted
    positive measurement.

    Missing frames remain missing. No height is calculated.
    """

    raw_heights = np.asarray(
        raw_heights,
        dtype=np.float64
    )

    raw_confidence_percent = np.asarray(
        raw_confidence_percent,
        dtype=np.float64
    )

    raw_wetting_scores = np.asarray(
        raw_wetting_scores,
        dtype=np.float64
    )

    if not (
        len(
            raw_heights
        )
        ==
        len(
            raw_confidence_percent
        )
        ==
        len(
            raw_wetting_scores
        )
    ):
        raise ValueError(
            "Onset-backtracking arrays must have equal length."
        )

    first_confirmed_index = int(
        first_confirmed_index
    )

    current_index = int(
        first_confirmed_index
    )

    current_height = float(
        first_confirmed_height
    )

    search_start = max(
        1,
        first_confirmed_index
        - int(
            TRACK_ONSET_BACKTRACK_FRAMES
        )
    )

    records_backwards = []

    consecutive_misses = 0

    found_any = False

    for frame_index in range(
        first_confirmed_index - 1,
        search_start - 1,
        -1
    ):

        height = float(
            raw_heights[
                frame_index
            ]
        )

        confidence = float(
            raw_confidence_percent[
                frame_index
            ]
        )

        wetting_score = float(
            raw_wetting_scores[
                frame_index
            ]
        )

        frame_gap = (
            current_index
            - frame_index
        )

        maximum_allowed_rise = (
            TRACK_ONSET_BACKTRACK_MAX_RISE_PER_FRAME_MM
            * frame_gap
            + TRACK_CONFIRM_TOLERANCE_MM
        )

        valid_candidate = (
            np.isfinite(
                height
            )
            and
            np.isfinite(
                confidence
            )
            and
            np.isfinite(
                wetting_score
            )
            and
            confidence
            >= TRACK_ONSET_BACKTRACK_MIN_CONFIDENCE_PERCENT
            and
            wetting_score
            >= TRACK_ONSET_BACKTRACK_MIN_WETTING_SCORE
            and
            height
            >= TRACK_ONSET_MIN_HEIGHT_MM
            and
            height
            <= current_height
            + 1e-9
            and
            (
                current_height
                - height
            )
            <= maximum_allowed_rise
        )

        if not valid_candidate:

            consecutive_misses += 1

            if (
                found_any
                and
                consecutive_misses
                > TRACK_ONSET_BACKTRACK_MAX_CONSECUTIVE_MISSES
            ):
                break

            continue

        records_backwards.append(
            {
                "anchor_index":
                    int(
                        frame_index
                    ),

                "height":
                    float(
                        height
                    ),

                "cluster_median":
                    np.nan,

                "support_indices":
                    np.asarray(
                        [
                            frame_index
                        ],
                        dtype=int
                    ),

                "support_count":
                    1,

                "type":
                    "onset_reviewed"
            }
        )

        current_index = int(
            frame_index
        )

        current_height = float(
            height
        )

        consecutive_misses = 0

        found_any = True

    records_backwards.reverse()

    return records_backwards




def build_final_measured_series(
    height_validated_mm,
    confirmation_records,
    raw_height_mm
):
    """
    Build the high-resolution FINAL MEASURED series.

    Every finite value in the returned series is an ACTUAL
    frame-derived value, except frame 0 which is the explicit
    experimental initial condition of 0 mm.

    Rules
    -----
    1. Keep every temporally validated actual frame measurement.
    2. Add retrospectively accepted onset measurements only from
       their exact original raw frame values.
    3. Never interpolate, reconstruct, smooth, forward-fill, or
       calculate missing temporal measurements.
    4. Unresolved frames remain np.nan.
    """

    height_validated_mm = np.asarray(
        height_validated_mm,
        dtype=np.float64
    )

    raw_height_mm = np.asarray(
        raw_height_mm,
        dtype=np.float64
    )

    if len(
        height_validated_mm
    ) != len(
        raw_height_mm
    ):
        raise ValueError(
            "Validated and raw height arrays must have equal length."
        )

    n = len(
        height_validated_mm
    )

    # Start from every actual temporally validated observation.
    final_measured = (
        height_validated_mm.copy()
    )

    source = np.full(
        n,
        "",
        dtype=object
    )

    validated_mask = np.isfinite(
        final_measured
    )

    source[
        validated_mask
    ] = "validated"

    # Explicit experimental initial condition.
    if n > 0:

        final_measured[
            0
        ] = float(
            TRACK_INITIAL_HEIGHT_MM
        )

        source[
            0
        ] = "initial_condition"

    # Retrospective onset review may recover an actual raw
    # observation that did not pass the normal temporal threshold.
    # Use the exact raw value from that frame; do not use a cluster
    # average or any derived/interpolated height.
    for record in confirmation_records:

        if record.get(
            "type"
        ) != "onset_reviewed":
            continue

        frame_index = int(
            record[
                "anchor_index"
            ]
        )

        if (
            frame_index < 0
            or
            frame_index >= n
        ):
            continue

        raw_value = float(
            raw_height_mm[
                frame_index
            ]
        )

        if not np.isfinite(
            raw_value
        ):
            continue

        final_measured[
            frame_index
        ] = raw_value

        source[
            frame_index
        ] = "onset_reviewed_raw"

    return (
        final_measured,
        source
    )


def build_monotonic_tracked_curve(
    time_s,
    validated_height_mm,
    validated_confidence_percent,
    raw_height_mm=None,
    raw_confidence_percent=None,
    raw_wetting_score=None
):
    """
    Measurement-only monotonic tracker.

    No temporal interpolation.
    No step holding.
    No reconstructed heights.

    `tracked` contains values only where an actual frame-derived
    measurement was accepted. Unresolved frames are np.nan.
    """

    time_s = np.asarray(
        time_s,
        dtype=np.float64
    )

    validated_height_mm = np.asarray(
        validated_height_mm,
        dtype=np.float64
    )

    validated_confidence_percent = np.asarray(
        validated_confidence_percent,
        dtype=np.float64
    )

    n = len(
        validated_height_mm
    )

    if n == 0:

        return (
            np.array(
                [],
                dtype=np.float64
            ),
            np.array(
                [],
                dtype=bool
            ),
            []
        )

    if not (
        len(
            time_s
        ) == n
        and
        len(
            validated_confidence_percent
        ) == n
    ):
        raise ValueError(
            "Tracking arrays must have equal length."
        )

    anchor_indices = [
        0
    ]

    anchor_heights = [
        float(
            TRACK_INITIAL_HEIGHT_MM
        )
    ]

    records = [
        {
            "anchor_index":
                0,

            "height":
                float(
                    TRACK_INITIAL_HEIGHT_MM
                ),

            "cluster_median":
                np.nan,

            "support_indices":
                np.asarray(
                    [
                        0
                    ],
                    dtype=int
                ),

            "support_count":
                1,

            "type":
                "initial"
        }
    ]

    current_height = float(
        TRACK_INITIAL_HEIGHT_MM
    )

    window_size = int(
        TRACK_CONFIRM_WINDOW
    )

    required = int(
        TRACK_CONFIRM_REQUIRED
    )

    if (
        window_size < 1
        or
        required < 1
        or
        required > window_size
    ):
        raise ValueError(
            "Invalid tracking confirmation settings."
        )

    last_start = (
        n
        - window_size
    )

    if last_start >= 0:

        for start in range(
            0,
            last_start + 1
        ):

            end = (
                start
                + window_size
            )

            confirmed = (
                find_confirmed_level_in_window(
                    validated_height_mm[
                        start:end
                    ],
                    validated_confidence_percent[
                        start:end
                    ],
                    np.arange(
                        start,
                        end,
                        dtype=int
                    ),
                    current_height
                )
            )

            if confirmed is None:
                continue

            confirmed_height = float(
                confirmed[
                    "height"
                ]
            )

            anchor_index = int(
                confirmed[
                    "anchor_index"
                ]
            )

            if (
                anchor_index
                <= anchor_indices[
                    -1
                ]
            ):
                continue

            if (
                confirmed_height
                < current_height
                + TRACK_MIN_RISE_MM
            ):
                continue

            if (
                confirmed_height
                - current_height
                >= TRACK_REVIEW_TRIGGER_MM
            ):

                reviewed = (
                    find_reviewed_anchors_between(
                        validated_height_mm,
                        validated_confidence_percent,
                        anchor_indices[
                            -1
                        ],
                        current_height,
                        anchor_index,
                        confirmed_height
                    )
                )

                for item in reviewed:

                    item_index = int(
                        item[
                            "anchor_index"
                        ]
                    )

                    item_height = float(
                        item[
                            "height"
                        ]
                    )

                    if (
                        item_index
                        <= anchor_indices[
                            -1
                        ]
                    ):
                        continue

                    if (
                        item_height
                        <= current_height
                        + TRACK_REVIEW_MIN_RISE_MM
                    ):
                        continue

                    if (
                        item_height
                        >= confirmed_height
                    ):
                        continue

                    anchor_indices.append(
                        item_index
                    )

                    anchor_heights.append(
                        item_height
                    )

                    records.append(
                        item
                    )

                    current_height = float(
                        item_height
                    )

            if (
                anchor_index
                <= anchor_indices[
                    -1
                ]
            ):
                continue

            if (
                confirmed_height
                < current_height
                + TRACK_MIN_RISE_MM
            ):
                continue

            anchor_indices.append(
                anchor_index
            )

            anchor_heights.append(
                confirmed_height
            )

            records.append(
                confirmed
            )

            current_height = float(
                confirmed_height
            )

    # --------------------------------------------------------
    # Retrospective onset review using actual raw measurements.
    # --------------------------------------------------------

    positive_records = [
        record
        for record
        in records

        if (
            int(
                record[
                    "anchor_index"
                ]
            )
            > 0
            and
            float(
                record[
                    "height"
                ]
            )
            > TRACK_INITIAL_HEIGHT_MM
            + 1e-9
        )
    ]

    if (
        positive_records
        and
        raw_height_mm is not None
        and
        raw_confidence_percent is not None
        and
        raw_wetting_score is not None
    ):

        first_positive = min(
            positive_records,
            key=lambda record:
            int(
                record[
                    "anchor_index"
                ]
            )
        )

        onset_records = (
            find_actual_onset_backtrack(
                raw_height_mm,
                raw_confidence_percent,
                raw_wetting_score,
                first_positive[
                    "anchor_index"
                ],
                first_positive[
                    "height"
                ]
            )
        )

        records.extend(
            onset_records
        )

    # --------------------------------------------------------
    # Deduplicate same-frame records.
    # --------------------------------------------------------

    priority = {
        "initial":
            0,

        "onset_reviewed":
            1,

        "reviewed":
            2,

        "confirmed":
            3
    }

    by_index = {}

    for record in records:

        index = int(
            record[
                "anchor_index"
            ]
        )

        if (
            index not in by_index
            or
            priority.get(
                record.get(
                    "type",
                    ""
                ),
                -1
            )
            >
            priority.get(
                by_index[
                    index
                ].get(
                    "type",
                    ""
                ),
                -1
            )
        ):
            by_index[
                index
            ] = record

    records = [
        by_index[
            index
        ]
        for index
        in sorted(
            by_index
        )
    ]

    # --------------------------------------------------------
    # Monotonic acceptance is enforced by rejecting inconsistent
    # observations. Measurements are never altered.
    # --------------------------------------------------------

    accepted_records = []

    last_height = -np.inf

    for record in records:

        index = int(
            record[
                "anchor_index"
            ]
        )

        height = float(
            record[
                "height"
            ]
        )

        if index == 0:

            accepted_records.append(
                record
            )

            last_height = height

            continue

        if (
            height
            + 1e-9
            < last_height
        ):
            continue

        accepted_records.append(
            record
        )

        last_height = height

    records = accepted_records

    tracked = np.full(
        n,
        np.nan,
        dtype=np.float64
    )

    anchor_mask = np.zeros(
        n,
        dtype=bool
    )

    for record in records:

        index = int(
            record[
                "anchor_index"
            ]
        )

        if (
            index < 0
            or
            index >= n
        ):
            continue

        tracked[
            index
        ] = float(
            record[
                "height"
            ]
        )

        anchor_mask[
            index
        ] = True

    return (
        tracked,
        anchor_mask,
        records
    )


def detect_rulers(
    image,
    log_callback=None
):
    """
    Detect left and right ruler boxes.

    The normal strict detector is tried first.

    If one ruler is missed, an adaptive fallback searches the
    corresponding side of the image for a broad, vertically
    persistent dark object. When the opposite ruler was detected
    normally, its width and vertical extent are used as additional
    geometric guidance.

    This function affects only static geometry calibration. It does
    not alter any capillary-rise measurement.
    """

    h, w = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    y0 = int(
        0.08
        * h
    )

    y1 = int(
        0.97
        * h
    )

    band = gray[
        y0:y1,
        :
    ]

    # --------------------------------------------------------
    # STRICT / ORIGINAL DETECTOR
    # --------------------------------------------------------

    dark_fraction = np.mean(
        band
        < RULER_DARK_THRESHOLD,
        axis=0
    )

    column_mask = (
        dark_fraction
        > RULER_MIN_DARK_FRACTION
    ).astype(
        np.uint8
    ) * 255

    close_width = odd_number(
        0.018
        * w
    )

    kernel = np.ones(
        (
            1,
            close_width
        ),
        dtype=np.uint8
    )

    column_mask = cv2.morphologyEx(
        column_mask.reshape(
            1,
            -1
        ),
        cv2.MORPH_CLOSE,
        kernel
    ).ravel()

    regions = []

    start = None

    for x in range(
        w
    ):

        active = (
            column_mask[
                x
            ]
            > 0
        )

        if (
            active
            and
            start is None
        ):

            start = x

        if (
            start is not None
            and
            (
                not active
                or
                x
                == w - 1
            )
        ):

            end = (
                x
                if active
                else x - 1
            )

            region_width = (
                end
                - start
                + 1
            )

            if (
                region_width
                > 0.04
                * w
            ):

                regions.append(
                    (
                        int(
                            start
                        ),
                        int(
                            end
                        )
                    )
                )

            start = None

    left_candidates = []

    right_candidates = []

    for region in regions:

        centre = (
            region[
                0
            ]
            + region[
                1
            ]
        ) / 2.0

        if (
            centre
            < 0.45
            * w
        ):

            left_candidates.append(
                region
            )

        elif (
            centre
            > 0.55
            * w
        ):

            right_candidates.append(
                region
            )

    def strict_region_score(
        region
    ):

        xs, xe = region

        mean_dark = float(
            np.mean(
                dark_fraction[
                    xs:
                    xe + 1
                ]
            )
        )

        region_width = (
            xe
            - xs
            + 1
        )

        return (
            mean_dark
            + 0.10
            * min(
                region_width
                / max(
                    1.0,
                    0.12
                    * w
                ),
                1.5
            )
        )

    left_region = (
        max(
            left_candidates,
            key=strict_region_score
        )
        if left_candidates
        else None
    )

    right_region = (
        max(
            right_candidates,
            key=strict_region_score
        )
        if right_candidates
        else None
    )

    # --------------------------------------------------------
    # BOX VERTICAL EXTENT
    # --------------------------------------------------------

    def make_box(
        region,
        reference_box=None
    ):

        xs, xe = region

        roi = gray[
            :,
            xs:
            xe + 1
        ]

        # More tolerant threshold than the strict horizontal
        # detector. White ruler ticks/numbers naturally reduce the
        # dark fraction of some rows.
        row_dark_fraction = np.mean(
            roi
            < max(
                145,
                RULER_DARK_THRESHOLD
                + 25
            ),
            axis=1
        )

        row_mask = (
            row_dark_fraction
            > 0.20
        ).astype(
            np.uint8
        ) * 255

        vertical_close = max(
            5,
            int(
                0.012
                * h
            )
        )

        row_mask = cv2.morphologyEx(
            row_mask.reshape(
                -1,
                1
            ),
            cv2.MORPH_CLOSE,
            np.ones(
                (
                    vertical_close,
                    1
                ),
                dtype=np.uint8
            )
        ).ravel()

        ys = np.where(
            row_mask
            > 0
        )[0]

        if len(
            ys
        ) == 0:

            if reference_box is None:

                raise RuntimeError(
                    "Could not determine ruler vertical extent."
                )

            top = int(
                reference_box[
                    1
                ]
            )

            bottom = int(
                reference_box[
                    1
                ]
                + reference_box[
                    3
                ]
                - 1
            )

        else:

            # Find the longest meaningful run rather than allowing
            # a small isolated dark object to define the box.
            runs = []

            run_start = int(
                ys[
                    0
                ]
            )

            previous = int(
                ys[
                    0
                ]
            )

            for y_value in ys[
                1:
            ]:

                y_value = int(
                    y_value
                )

                if (
                    y_value
                    >
                    previous + 1
                ):

                    runs.append(
                        (
                            run_start,
                            previous
                        )
                    )

                    run_start = (
                        y_value
                    )

                previous = (
                    y_value
                )

            runs.append(
                (
                    run_start,
                    previous
                )
            )

            top, bottom = max(
                runs,
                key=lambda run:
                run[
                    1
                ]
                - run[
                    0
                ]
            )

            # If the opposite ruler is already reliable, use its
            # vertical span as guidance. The two physical rulers in
            # this setup are mounted at essentially the same height.
            if reference_box is not None:

                reference_top = int(
                    reference_box[
                        1
                    ]
                )

                reference_bottom = int(
                    reference_box[
                        1
                    ]
                    + reference_box[
                        3
                    ]
                    - 1
                )

                top = min(
                    int(
                        top
                    ),
                    reference_top
                )

                bottom = max(
                    int(
                        bottom
                    ),
                    reference_bottom
                )

        top = int(
            np.clip(
                top,
                0,
                h - 1
            )
        )

        bottom = int(
            np.clip(
                bottom,
                top,
                h - 1
            )
        )

        box_width = int(
            xe
            - xs
            + 1
        )

        box_height = int(
            bottom
            - top
            + 1
        )

        return (
            int(
                xs
            ),
            int(
                top
            ),
            int(
                box_width
            ),
            int(
                box_height
            ),
            int(
                box_width
                * box_height
            )
        )

    # First construct any ruler that the strict detector found.
    left_box = None

    right_box = None

    if left_region is not None:

        try:

            left_box = make_box(
                left_region
            )

        except RuntimeError:

            left_box = None

    if right_region is not None:

        try:

            right_box = make_box(
                right_region
            )

        except RuntimeError:

            right_box = None

    # --------------------------------------------------------
    # ADAPTIVE HORIZONTAL FALLBACK
    # --------------------------------------------------------

    def find_fallback_region(
        side,
        reference_box=None
    ):

        # Ignore the very top and very bottom because clamps,
        # reflections and the reservoir can make those sections
        # unrepresentative. A ruler remains dark through most of
        # the central height.
        fallback_y0 = int(
            0.15
            * h
        )

        fallback_y1 = int(
            0.93
            * h
        )

        fallback_band = gray[
            fallback_y0:
            fallback_y1,
            :
        ]

        strict_column_dark = np.mean(
            fallback_band
            < RULER_DARK_THRESHOLD,
            axis=0
        )

        moderate_column_dark = np.mean(
            fallback_band
            < max(
                155,
                RULER_DARK_THRESHOLD
                + 45
            ),
            axis=0
        )

        # Give most weight to truly black pixels, but tolerate a
        # ruler whose apparent black level was lifted by exposure.
        metric = (
            0.65
            * strict_column_dark
            +
            0.35
            * moderate_column_dark
        )

        if side == "left":

            search_x0 = int(
                0.05
                * w
            )

            search_x1 = int(
                0.47
                * w
            )

        elif side == "right":

            search_x0 = int(
                0.53
                * w
            )

            search_x1 = int(
                0.95
                * w
            )

        else:

            raise ValueError(
                "side must be left or right"
            )

        if reference_box is not None:

            reference_width = int(
                reference_box[
                    2
                ]
            )

        else:

            reference_width = int(
                0.16
                * w
            )

        minimum_width = max(
            35,
            int(
                0.075
                * w
            )
        )

        maximum_width = min(
            int(
                0.23
                * w
            ),
            search_x1
            - search_x0
        )

        width_candidates = sorted(
            set(
                [
                    int(
                        np.clip(
                            reference_width
                            * factor,
                            minimum_width,
                            maximum_width
                        )
                    )
                    for factor
                    in (
                        0.72,
                        0.85,
                        1.00,
                        1.15,
                        1.30
                    )
                ]
            )
        )

        cumulative = np.concatenate(
            (
                np.asarray(
                    [
                        0.0
                    ],
                    dtype=np.float64
                ),
                np.cumsum(
                    metric,
                    dtype=np.float64
                )
            )
        )

        preliminary = []

        for candidate_width in width_candidates:

            latest_start = (
                search_x1
                - candidate_width
            )

            if (
                latest_start
                <= search_x0
            ):
                continue

            starts = np.arange(
                search_x0,
                latest_start + 1,
                dtype=int
            )

            window_scores = (
                cumulative[
                    starts
                    + candidate_width
                ]
                - cumulative[
                    starts
                ]
            ) / float(
                candidate_width
            )

            top_count = min(
                20,
                len(
                    starts
                )
            )

            if top_count <= 0:
                continue

            if (
                top_count
                == len(
                    starts
                )
            ):

                best_positions = np.arange(
                    len(
                        starts
                    )
                )

            else:

                best_positions = np.argpartition(
                    window_scores,
                    -top_count
                )[
                    -top_count:
                ]

            for position in best_positions:

                xs = int(
                    starts[
                        position
                    ]
                )

                xe = int(
                    xs
                    + candidate_width
                    - 1
                )

                preliminary.append(
                    (
                        float(
                            window_scores[
                                position
                            ]
                        ),
                        xs,
                        xe
                    )
                )

        if not preliminary:

            return None

        preliminary.sort(
            key=lambda item:
            item[
                0
            ],
            reverse=True
        )

        best = None

        for (
            horizontal_score,
            xs,
            xe
        ) in preliminary[
            :60
        ]:

            candidate_roi = gray[
                fallback_y0:
                fallback_y1,
                xs:
                xe + 1
            ]

            row_dark = np.mean(
                candidate_roi
                < max(
                    155,
                    RULER_DARK_THRESHOLD
                    + 45
                ),
                axis=1
            )

            if len(
                row_dark
            ) < 8:
                continue

            # The ruler should remain dark in many separated
            # vertical sections. A clamp or isolated fixture tends
            # to be dark only in part of the height.
            sections = np.array_split(
                row_dark,
                8
            )

            section_scores = np.asarray(
                [
                    float(
                        np.mean(
                            section
                        )
                    )
                    for section
                    in sections
                    if len(
                        section
                    ) > 0
                ],
                dtype=np.float64
            )

            continuity_score = float(
                np.median(
                    section_scores
                )
            )

            persistent_sections = float(
                np.mean(
                    section_scores
                    > 0.28
                )
            )

            # Use the opposite ruler's width if available. This
            # helps distinguish the broad ruler from narrow black
            # fixture posts.
            if reference_box is not None:

                width_ratio = (
                    (
                        xe
                        - xs
                        + 1
                    )
                    /
                    max(
                        1.0,
                        float(
                            reference_box[
                                2
                            ]
                        )
                    )
                )

                width_score = float(
                    np.exp(
                        -abs(
                            np.log(
                                max(
                                    1e-6,
                                    width_ratio
                                )
                            )
                        )
                    )
                )

            else:

                width_score = 1.0

            total_score = (
                0.48
                * horizontal_score
                +
                0.32
                * continuity_score
                +
                0.12
                * persistent_sections
                +
                0.08
                * width_score
            )

            candidate = (
                total_score,
                xs,
                xe,
                horizontal_score,
                continuity_score,
                persistent_sections
            )

            if (
                best is None
                or
                candidate[
                    0
                ]
                > best[
                    0
                ]
            ):

                best = candidate

        if best is None:

            return None

        (
            total_score,
            xs,
            xe,
            horizontal_score,
            continuity_score,
            persistent_sections
        ) = best

        # Intentionally permissive because this is already limited
        # to one side of the image and usually guided by the other
        # ruler's dimensions.
        if (
            total_score
            < 0.28
        ):

            return None

        emit_log(
            log_callback,
            (
                f"Adaptive {side} ruler fallback: "
                f"x={xs}:{xe}, "
                f"score={total_score:.3f}, "
                f"dark={horizontal_score:.3f}, "
                f"continuity={continuity_score:.3f}."
            )
        )

        return (
            int(
                xs
            ),
            int(
                xe
            )
        )

    # If only one side was found strictly, use it to guide the
    # missing side.
    if left_box is None:

        fallback_left = find_fallback_region(
            "left",
            reference_box=right_box
        )

        if fallback_left is not None:

            left_box = make_box(
                fallback_left,
                reference_box=right_box
            )

    if right_box is None:

        fallback_right = find_fallback_region(
            "right",
            reference_box=left_box
        )

        if fallback_right is not None:

            right_box = make_box(
                fallback_right,
                reference_box=left_box
            )

    # If both strict detections failed, the first fallback had no
    # reference. Give the second pass the newly found box.
    if left_box is None:

        fallback_left = find_fallback_region(
            "left",
            reference_box=right_box
        )

        if fallback_left is not None:

            left_box = make_box(
                fallback_left,
                reference_box=right_box
            )

    if right_box is None:

        fallback_right = find_fallback_region(
            "right",
            reference_box=left_box
        )

        if fallback_right is not None:

            right_box = make_box(
                fallback_right,
                reference_box=left_box
            )

    if left_box is None:

        raise RuntimeError(
            "Could not detect left ruler."
        )

    if right_box is None:

        raise RuntimeError(
            "Could not detect right ruler."
        )

    return (
        left_box,
        right_box
    )




def detect_rulers_by_tick_pattern(
    image,
    log_callback=None
):
    """
    Detect ruler boxes from their repeated horizontal tick pattern.

    This fallback is deliberately different from the normal dark-
    rectangle detector. A ruler is identified by:
        - strong repeated horizontal image gradients;
        - a broad vertically persistent rectangular region;
        - an expected width range;
        - side-specific position.

    This helps reject dark clamps and fixtures, which may be very
    black but do not contain the same dense horizontal tick pattern.

    Static geometry only. No height-vs-time data are created.
    """

    h, w = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    analysis_y0 = int(
        0.10
        * h
    )

    analysis_y1 = int(
        0.94
        * h
    )

    analysis = gray[
        analysis_y0:
        analysis_y1,
        :
    ]

    grad_y = np.abs(
        cv2.Sobel(
            analysis,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )
    )

    # Adaptive but with a useful minimum. On the supplied failing
    # Top2 screenshots this is about 48 gray-level gradient units.
    edge_threshold = max(
        35.0,
        float(
            np.percentile(
                grad_y,
                82
            )
        )
    )

    horizontal_edge = (
        grad_y
        > edge_threshold
    ).astype(
        np.float32
    )

    marker_width_fractions = (
        0.13,
        0.15,
        0.17,
        0.19,
        0.21
    )

    def score_window(
        x0,
        width
    ):

        x1 = (
            x0
            + width
        )

        edge_roi = horizontal_edge[
            :,
            x0:x1
        ]

        gray_roi = analysis[
            :,
            x0:x1
        ]

        if (
            edge_roi.size == 0
            or
            gray_roi.size == 0
        ):
            return -np.inf

        # Overall density of strong horizontal gradients.
        horizontal_density = float(
            np.mean(
                edge_roi
            )
        )

        # Fraction of rows that contain a substantial horizontal
        # line segment. The repeated ruler ticks score very highly.
        row_coverage = np.mean(
            edge_roi,
            axis=1
        )

        repeated_line_fraction = float(
            np.mean(
                row_coverage
                > 0.16
            )
        )

        # Black ruler background.
        dark_fraction = float(
            np.mean(
                gray_roi
                < 150
            )
        )

        # Rulers have strong black/white contrast due to the ticks,
        # major lines and digits.
        contrast_score = min(
            float(
                np.std(
                    gray_roi
                )
                / 128.0
            ),
            1.5
        )

        return (
            0.35
            * horizontal_density
            * 5.0
            +
            0.30
            * repeated_line_fraction
            * 3.0
            +
            0.20
            * dark_fraction
            +
            0.15
            * contrast_score
        )

    def find_horizontal_region(
        side
    ):

        if side == "left":

            search_x0 = int(
                0.05
                * w
            )

            search_x1 = int(
                0.48
                * w
            )

        elif side == "right":

            search_x0 = int(
                0.52
                * w
            )

            search_x1 = int(
                0.95
                * w
            )

        else:

            raise ValueError(
                "side must be left or right"
            )

        coarse_step = max(
            1,
            int(
                0.004
                * w
            )
        )

        best = None

        for width_fraction in marker_width_fractions:

            candidate_width = int(
                round(
                    width_fraction
                    * w
                )
            )

            if candidate_width < 20:
                continue

            last_start = (
                search_x1
                - candidate_width
            )

            if last_start <= search_x0:
                continue

            for x0 in range(
                search_x0,
                last_start + 1,
                coarse_step
            ):

                score = score_window(
                    x0,
                    candidate_width
                )

                if (
                    best is None
                    or
                    score
                    > best[
                        0
                    ]
                ):

                    best = (
                        float(
                            score
                        ),
                        int(
                            x0
                        ),
                        int(
                            candidate_width
                        )
                    )

        if best is None:

            raise RuntimeError(
                f"Could not find {side} ruler tick-pattern region."
            )

        best_score, best_x0, best_width = best

        # Pixel-level refinement around the best coarse position.
        refinement_radius = max(
            2,
            coarse_step
        )

        refined_start = max(
            search_x0,
            best_x0
            - refinement_radius
        )

        refined_end = min(
            search_x1
            - best_width,
            best_x0
            + refinement_radius
        )

        for x0 in range(
            refined_start,
            refined_end + 1
        ):

            score = score_window(
                x0,
                best_width
            )

            if score > best_score:

                best_score = float(
                    score
                )

                best_x0 = int(
                    x0
                )

        emit_log(
            log_callback,
            (
                f"Tick-pattern {side} ruler candidate: "
                f"x={best_x0}:"
                f"{best_x0 + best_width - 1}, "
                f"score={best_score:.3f}."
            )
        )

        return (
            best_x0,
            best_x0
            + best_width
            - 1,
            best_score
        )

    def estimate_vertical_box(
        region
    ):

        x0, x1, _ = region

        roi = gray[
            :,
            x0:
            x1 + 1
        ]

        local_grad_y = np.abs(
            cv2.Sobel(
                roi,
                cv2.CV_32F,
                0,
                1,
                ksize=3
            )
        )

        row_edge_fraction = np.mean(
            local_grad_y
            > edge_threshold,
            axis=1
        )

        row_dark_fraction = np.mean(
            roi
            < 160,
            axis=1
        )

        presence = (
            (
                row_edge_fraction
                > 0.04
            )
            |
            (
                row_dark_fraction
                > 0.50
            )
        ).astype(
            np.uint8
        ) * 255

        close_height = max(
            5,
            int(
                0.015
                * h
            )
        )

        presence = cv2.morphologyEx(
            presence.reshape(
                -1,
                1
            ),
            cv2.MORPH_CLOSE,
            np.ones(
                (
                    close_height,
                    1
                ),
                dtype=np.uint8
            )
        ).ravel()

        ys = np.where(
            presence
            > 0
        )[0]

        runs = []

        if len(
            ys
        ) > 0:

            run_start = int(
                ys[
                    0
                ]
            )

            previous = int(
                ys[
                    0
                ]
            )

            for y_value in ys[
                1:
            ]:

                y_value = int(
                    y_value
                )

                if (
                    y_value
                    > previous + 1
                ):

                    runs.append(
                        (
                            run_start,
                            previous
                        )
                    )

                    run_start = y_value

                previous = y_value

            runs.append(
                (
                    run_start,
                    previous
                )
            )

        long_runs = [
            run
            for run
            in runs
            if (
                run[
                    1
                ]
                - run[
                    0
                ]
                >= 0.40
                * h
            )
        ]

        if long_runs:

            top, bottom = max(
                long_runs,
                key=lambda run:
                run[
                    1
                ]
                - run[
                    0
                ]
            )

        else:

            # Safe static-geometry fallback. The later major-line
            # validation still has to succeed before this frame can
            # be accepted.
            top = int(
                0.08
                * h
            )

            bottom = int(
                0.97
                * h
            )

        width = int(
            x1
            - x0
            + 1
        )

        box_height = int(
            bottom
            - top
            + 1
        )

        return (
            int(
                x0
            ),
            int(
                top
            ),
            int(
                width
            ),
            int(
                box_height
            ),
            int(
                width
                * box_height
            )
        )

    left_region = find_horizontal_region(
        "left"
    )

    right_region = find_horizontal_region(
        "right"
    )

    left_box = estimate_vertical_box(
        left_region
    )

    right_box = estimate_vertical_box(
        right_region
    )

    # The two rulers are mounted at essentially the same vertical
    # position. Use the common overlapping span, when sufficiently
    # large, so both major-line detectors analyse matching heights.
    common_top = max(
        left_box[
            1
        ],
        right_box[
            1
        ]
    )

    common_bottom = min(
        left_box[
            1
        ]
        + left_box[
            3
        ]
        - 1,
        right_box[
            1
        ]
        + right_box[
            3
        ]
        - 1
    )

    if (
        common_bottom
        - common_top
        >= 0.40
        * h
    ):

        left_box = (
            left_box[
                0
            ],
            int(
                common_top
            ),
            left_box[
                2
            ],
            int(
                common_bottom
                - common_top
                + 1
            ),
            int(
                left_box[
                    2
                ]
                * (
                    common_bottom
                    - common_top
                    + 1
                )
            )
        )

        right_box = (
            right_box[
                0
            ],
            int(
                common_top
            ),
            right_box[
                2
            ],
            int(
                common_bottom
                - common_top
                + 1
            ),
            int(
                right_box[
                    2
                ]
                * (
                    common_bottom
                    - common_top
                    + 1
                )
            )
        )

    return (
        left_box,
        right_box
    )


def find_ruler_reference_frame(
    cap,
    first_frame,
    fps,
    total_frames,
    log_callback=None
):
    """
    Find an early frame with a fully valid ruler calibration.

    Validation is now end-to-end:
        1. detect ruler boxes;
        2. detect major lines on both rulers;
        3. pair the major lines;
        4. verify that a 15-line 75 -> 5 sequence exists.

    If the normal ruler detector produces plausible boxes but the
    major-line calibration fails, the tick-pattern detector is tried
    on that same frame before moving to another early frame.

    Only static geometry is taken from a later reference frame.
    Temporal wetting analysis continues to use frame 0.
    """

    candidate_indices = [
        0,
        1,
        2,
        5,
        10,
        int(
            round(
                0.25
                * fps
            )
        ),
        int(
            round(
                0.50
                * fps
            )
        ),
        int(
            round(
                0.75
                * fps
            )
        ),
        int(
            round(
                1.00
                * fps
            )
        ),
        int(
            round(
                1.50
                * fps
            )
        ),
        int(
            round(
                2.00
                * fps
            )
        ),
        int(
            round(
                3.00
                * fps
            )
        )
    ]

    candidate_indices = sorted(
        set(
            [
                int(
                    np.clip(
                        index,
                        0,
                        max(
                            0,
                            total_frames
                            - 1
                        )
                    )
                )
                for index
                in candidate_indices
            ]
        )
    )

    last_error = None

    def validate_boxes(
        frame,
        frame_index,
        left_ruler,
        right_ruler,
        detector_name
    ):

        left_major_marks = (
            detect_major_ruler_marks(
                frame,
                left_ruler,
                "left"
            )
        )

        right_major_marks = (
            detect_major_ruler_marks(
                frame,
                right_ruler,
                "right"
            )
        )

        major_pairs = pair_major_marks(
            left_major_marks,
            right_major_marks,
            frame.shape[
                0
            ]
        )

        emit_log(
            log_callback,
            (
                f"{detector_name} ruler validation on "
                f"frame {frame_index}: "
                f"left major={len(left_major_marks)}, "
                f"right major={len(right_major_marks)}, "
                f"paired={len(major_pairs)}."
            )
        )

        if len(
            major_pairs
        ) < 15:

            raise RuntimeError(
                (
                    f"{detector_name} ruler boxes produced "
                    f"only {len(major_pairs)} paired major "
                    f"lines."
                )
            )

        # This verifies the actual 75 -> 5 sequence, not merely
        # the number of detected lines.
        select_top_down_major_scale(
            major_pairs
        )

        return True

    for frame_index in candidate_indices:

        if frame_index == 0:

            frame = first_frame

        else:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index
            )

            success, frame = cap.read()

            if not success:
                continue

        # ----------------------------------------------------
        # A. Normal / existing detector first
        # ----------------------------------------------------

        try:

            (
                left_ruler,
                right_ruler
            ) = detect_rulers(
                frame,
                log_callback=log_callback
            )

            validate_boxes(
                frame,
                frame_index,
                left_ruler,
                right_ruler,
                "Standard"
            )

            if frame_index > 0:

                emit_log(
                    log_callback,
                    (
                        f"Ruler geometry recovered from "
                        f"frame {frame_index} "
                        f"({frame_index / fps:.3f} s) "
                        f"instead of frame 0."
                    )
                )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            return (
                frame,
                int(
                    frame_index
                ),
                left_ruler,
                right_ruler
            )

        except RuntimeError as error:

            last_error = error

        # ----------------------------------------------------
        # B. Tick-pattern detector on the same frame
        # ----------------------------------------------------

        try:

            (
                left_ruler,
                right_ruler
            ) = detect_rulers_by_tick_pattern(
                frame,
                log_callback=log_callback
            )

            validate_boxes(
                frame,
                frame_index,
                left_ruler,
                right_ruler,
                "Tick-pattern"
            )

            emit_log(
                log_callback,
                (
                    f"Tick-pattern ruler fallback accepted "
                    f"on frame {frame_index} "
                    f"({frame_index / fps:.3f} s)."
                )
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            return (
                frame,
                int(
                    frame_index
                ),
                left_ruler,
                right_ruler
            )

        except RuntimeError as error:

            last_error = error

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    raise RuntimeError(
        (
            "Could not obtain a valid ruler calibration "
            "from any tested early frame. "
            f"Last error: {last_error}"
        )
    )


def estimate_major_spacing(
    marks,
    ruler_box
):

    x, y, w, h, _ = ruler_box

    marks = sorted(
        marks
    )

    if len(marks) < 2:

        raise RuntimeError(
            "Not enough major marks."
        )

    upper_limit = (
        y + 0.82 * h
    )

    upper_marks = [
        value
        for value in marks
        if value <= upper_limit
    ]

    if len(upper_marks) < 3:

        upper_marks = marks[
            :max(
                3,
                int(
                    0.80 * len(marks)
                )
            )
        ]

    differences = np.diff(
        upper_marks
    ).astype(
        np.float32
    )

    minimum = (
        0.02 * h
    )

    maximum = (
        0.10 * h
    )

    plausible = differences[
        (
            differences > minimum
        )
        &
        (
            differences < maximum
        )
    ]

    if len(plausible) > 0:

        return float(
            np.median(
                plausible
            )
        )

    return float(
        np.median(
            differences
        )
    )


# ============================================================
# REGULARISE MAJOR MARKS
# ============================================================

def regularize_major_marks(
    marks,
    ruler_box
):

    marks = sorted(
        marks
    )

    if len(marks) < 3:
        return marks

    spacing = estimate_major_spacing(
        marks,
        ruler_box
    )

    cleaned = [
        marks[0]
    ]

    for current in marks[1:]:

        previous = (
            cleaned[-1]
        )

        gap = (
            current
            - previous
        )

        if gap < (
            0.55 * spacing
        ):
            continue

        if gap < (
            1.55 * spacing
        ):

            cleaned.append(
                current
            )

            continue

        intervals = max(
            1,
            int(
                round(
                    gap / spacing
                )
            )
        )

        if intervals >= 2:

            for k in range(
                1,
                intervals
            ):

                reconstructed = int(
                    round(
                        previous
                        + gap
                        * k
                        / intervals
                    )
                )

                cleaned.append(
                    reconstructed
                )

        cleaned.append(
            current
        )

    return cleaned


# ============================================================
# DETECT MAJOR RULER MARKS
# ============================================================

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

    enhanced = clahe.apply(
        roi
    )

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
        -3
    )

    if side == "left":

        band_x0 = int(
            0.58 * w
        )

        band_x1 = int(
            0.82 * w
        )

    elif side == "right":

        band_x0 = int(
            0.18 * w
        )

        band_x1 = int(
            0.42 * w
        )

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
        int(
            0.05 * w
        )
    )

    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                kernel_width,
                1
            )
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
    ).astype(
        np.float32
    )

    smoothing = odd_number(
        0.004 * h
    )

    row_strength = smooth_1d(
        row_strength,
        smoothing
    )

    margin = max(
        5,
        int(
            0.005 * h
        )
    )

    row_strength[
        :margin
    ] = 0

    row_strength[
        -margin:
    ] = 0

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
        0.18
        * (
            band_x1
            - band_x0
        )
    )

    min_distance = max(
        10,
        int(
            MAJOR_MIN_DISTANCE_REL
            * h
        )
    )

    peaks = find_peaks_1d(
        row_strength,
        min_distance,
        threshold
    )

    marks = [
        int(
            y + p
        )
        for p in peaks
    ]

    return regularize_major_marks(
        marks,
        ruler_box
    )


# ============================================================
# PAIR LEFT / RIGHT MAJOR MARKS
# ============================================================

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

            difference = abs(
                y_left - y_right
            )

            candidates.append(
                (
                    difference,
                    j,
                    y_right
                )
            )

        if not candidates:
            continue

        difference, j, y_right = min(
            candidates,
            key=lambda item:
            item[0]
        )

        if difference <= tolerance:

            paired.append(
                (
                    int(y_left),
                    int(y_right)
                )
            )

            used_right.add(
                j
            )

    paired.sort(
        key=lambda pair:
        (
            pair[0]
            + pair[1]
        ) / 2.0
    )

    return paired


# ============================================================
# TRUE 75 -> 5 MAJOR SEQUENCE
# ============================================================

def select_top_down_major_scale(
    major_pairs
):

    expected_values = list(
        range(
            TOP_MAJOR_VALUE,
            BOTTOM_MAJOR_VALUE - 1,
            -MAJOR_STEP
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
            dtype=np.float32
        )

        right_values = np.asarray(
            [
                pair[1]
                for pair in window
            ],
            dtype=np.float32
        )

        left_steps = np.diff(
            left_values
        )

        right_steps = np.diff(
            right_values
        )

        if (
            np.any(left_steps <= 0)
            or
            np.any(right_steps <= 0)
        ):
            continue

        median_left = float(
            np.median(
                left_steps
            )
        )

        median_right = float(
            np.median(
                right_steps
            )
        )

        left_error = np.mean(
            np.abs(
                left_steps
                - median_left
            )
            /
            max(
                median_left,
                1.0
            )
        )

        right_error = np.mean(
            np.abs(
                right_steps
                - median_right
            )
            /
            max(
                median_right,
                1.0
            )
        )

        pair_error = np.mean(
            np.abs(
                (
                    left_steps
                    /
                    max(
                        median_left,
                        1.0
                    )
                )
                -
                (
                    right_steps
                    /
                    max(
                        median_right,
                        1.0
                    )
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
        best_start
        + expected_count
    ]

    major_scale = []

    for value, pair in zip(
        expected_values,
        selected
    ):

        major_scale.append(
            {
                "value":
                    int(value),

                "left_y":
                    float(pair[0]),

                "right_y":
                    float(pair[1])
            }
        )

    return major_scale


# ============================================================
# CALCULATE ZERO
# ============================================================

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
            -
            recent[i]["left_y"]
        )

        right_steps.append(
            recent[i + 1]["right_y"]
            -
            recent[i]["right_y"]
        )

    left_spacing = float(
        np.median(
            left_steps
        )
    )

    right_spacing = float(
        np.median(
            right_steps
        )
    )

    five = (
        major_scale[-1]
    )

    return {
        "value":
            0,

        "left_y":
            five["left_y"]
            + left_spacing,

        "right_y":
            five["right_y"]
            + right_spacing
    }


# ============================================================
# CALCULATE 76 -> 79
# ============================================================

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
            -
            early[i]["left_y"]
        )

        right_steps.append(
            early[i + 1]["right_y"]
            -
            early[i]["right_y"]
        )

    left_major_spacing = float(
        np.median(
            left_steps
        )
    )

    right_major_spacing = float(
        np.median(
            right_steps
        )
    )

    left_unit = (
        left_major_spacing
        / MAJOR_STEP
    )

    right_unit = (
        right_major_spacing
        / MAJOR_STEP
    )

    line_75 = (
        major_scale[0]
    )

    result = []

    for value in range(
        TOP_MAJOR_VALUE + 1,
        TOP_VISIBLE_VALUE + 1
    ):

        delta = (
            value
            - TOP_MAJOR_VALUE
        )

        result.append(
            {
                "value":
                    int(value),

                "left_y":
                    line_75["left_y"]
                    - delta
                    * left_unit,

                "right_y":
                    line_75["right_y"]
                    - delta
                    * right_unit
            }
        )

    return result


# ============================================================
# COMPLETE SCALE
# ============================================================

def generate_measurement_lines(
    major_scale
):

    zero = calculate_zero_anchor(
        major_scale
    )

    anchors = (
        list(major_scale)
        +
        [
            zero
        ]
    )

    lines = []

    for i in range(
        len(anchors) - 1
    ):

        top = anchors[i]

        bottom = anchors[
            i + 1
        ]

        top_value = top[
            "value"
        ]

        bottom_value = bottom[
            "value"
        ]

        difference = (
            top_value
            - bottom_value
        )

        for j in range(
            difference
        ):

            alpha = (
                j
                / difference
            )

            value = (
                top_value
                - j
            )

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
                    "value":
                        int(value),

                    "left_y":
                        float(left_y),

                    "right_y":
                        float(right_y)
                }
            )

    lines.append(
        {
            "value":
                0,

            "left_y":
                float(
                    zero["left_y"]
                ),

            "right_y":
                float(
                    zero["right_y"]
                )
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
# COARSE SAMPLE DETECTION
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
    ).astype(
        np.float32
    )

    search_x0 = int(
        SAMPLE_SEARCH_LEFT
        * w
    )

    search_x1 = int(
        SAMPLE_SEARCH_RIGHT
        * w
    )

    red_score = (
        (R - G)
        + 0.20
        * (R - B)
        - 0.20
        * gray
    )

    roi_score = red_score[
        :,
        search_x0:
        search_x1
    ]

    score_u8 = cv2.normalize(
        roi_score,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(
        np.uint8
    )

    _, mask = cv2.threshold(
        score_u8,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    k = odd_number(
        0.004 * w
    )

    kernel = np.ones(
        (
            k,
            k
        ),
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
        centroids
    ) = cv2.connectedComponentsWithStats(
        mask
    )

    target_x = (
        search_x1
        - search_x0
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

        cx, cy = (
            centroids[label]
        )

        if bh < 0.35 * h:
            continue

        if area < (
            0.0005
            * h
            * w
        ):
            continue

        centre_error = abs(
            cx
            - target_x
        )

        value = (
            centre_error
            - 0.02
            * bh
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
        labels
        == best_label
    )

    ys = []

    left_xs = []

    right_xs = []

    for yy in range(h):

        xs_local = np.where(
            sample_mask[
                yy
            ]
        )[0]

        if len(xs_local) < 4:
            continue

        ys.append(
            yy
        )

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

    if len(ys) < (
        0.05 * h
    ):

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

    left_line = (
        float(a_left),
        float(b_left)
    )

    right_line = (
        float(a_right),
        float(b_right)
    )

    y_min = int(
        np.percentile(
            ys,
            2
        )
    )

    y_max = int(
        np.percentile(
            ys,
            98
        )
    )

    return (
        left_line,
        right_line,
        y_min,
        y_max
    )


# ============================================================
# ROBUST SAMPLE LINE FIT
# ============================================================

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
        (
            xs,
            ys
        )
    ).astype(
        np.float32
    )

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_HUBER,
        0,
        0.01,
        0.01
    ).flatten()

    if abs(vy) < 1e-8:

        raise RuntimeError(
            "Invalid sample edge fit."
        )

    a = (
        vx / vy
    )

    b = (
        x0
        - a
        * y0
    )

    predicted = (
        a
        * ys
        + b
    )

    residuals = np.abs(
        xs
        - predicted
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
        3.0
        * 1.4826
        * mad
    )

    keep = (
        residuals
        <= threshold
    )

    if np.sum(keep) >= 20:

        points = np.column_stack(
            (
                xs[keep],
                ys[keep]
            )
        ).astype(
            np.float32
        )

        vx, vy, x0, y0 = cv2.fitLine(
            points,
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01
        ).flatten()

        if abs(vy) > 1e-8:

            a = (
                vx / vy
            )

            b = (
                x0
                - a
                * y0
            )

    return (
        float(a),
        float(b)
    )


# ============================================================
# PRECISE SAMPLE EDGES
# ============================================================

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
    ).astype(
        np.float32
    )

    L = lab[:, :, 0]

    A = lab[:, :, 1]

    score = (
        A
        - 0.12
        * L
    )

    blur_x = odd_number(
        SAMPLE_EDGE_BLUR_X_REL
        * w
    )

    blur_y = odd_number(
        SAMPLE_EDGE_BLUR_Y_REL
        * h
    )

    score = cv2.GaussianBlur(
        score,
        (
            blur_x,
            blur_y
        ),
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
            SAMPLE_EDGE_ROW_STEP_REL
            * h
        )
    )

    detected_y = []

    detected_left = []

    detected_right = []

    detected_strength = []

    for y in range(
        max(
            sample_y_min,
            5
        ),
        min(
            sample_y_max,
            h - 5
        ),
        row_step
    ):

        coarse_x_left = (
            coarse_left[0]
            * y
            + coarse_left[1]
        )

        coarse_x_right = (
            coarse_right[0]
            * y
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
                - 0.55
                * coarse_width
            )
        )

        left_end = int(
            round(
                centre_x
                - 0.02
                * coarse_width
            )
        )

        right_start = int(
            round(
                centre_x
                + 0.02
                * coarse_width
            )
        )

        right_end = int(
            round(
                centre_x
                + 0.55
                * coarse_width
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
            left_start:
            left_end + 1
        ]

        right_gradient = gradient_x[
            y,
            right_start:
            right_end + 1
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
            right_edge
            - left_edge
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

        detected_y.append(
            y
        )

        detected_left.append(
            left_edge
        )

        detected_right.append(
            right_edge
        )

        detected_strength.append(
            min(
                left_strength,
                right_strength
            )
        )

    minimum_required = max(
        30,
        int(
            0.03 * h
        )
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

    detected_y = (
        detected_y[
            keep
        ]
    )

    detected_left = (
        detected_left[
            keep
        ]
    )

    detected_right = (
        detected_right[
            keep
        ]
    )

    widths = (
        detected_right
        - detected_left
    )

    median_width = np.median(
        widths
    )

    width_mad = np.median(
        np.abs(
            widths
            - median_width
        )
    ) + 1e-6

    width_tolerance = max(
        3.0,
        3.5
        * 1.4826
        * width_mad
    )

    keep = (
        np.abs(
            widths
            - median_width
        )
        <= width_tolerance
    )

    detected_y = (
        detected_y[
            keep
        ]
    )

    detected_left = (
        detected_left[
            keep
        ]
    )

    detected_right = (
        detected_right[
            keep
        ]
    )

    if len(detected_y) < 20:

        raise RuntimeError(
            "Too many sample-edge points rejected."
        )

    left_line = robust_fit_x_as_function_of_y(
        detected_left,
        detected_y
    )

    right_line = robust_fit_x_as_function_of_y(
        detected_right,
        detected_y
    )

    refined_y_min = int(
        np.percentile(
            detected_y,
            2
        )
    )

    refined_y_max = int(
        np.percentile(
            detected_y,
            98
        )
    )

    return (
        left_line,
        right_line,
        refined_y_min,
        refined_y_max
    )


# ============================================================
# SAMPLE DETECTOR
# ============================================================

def detect_sample_edges(
    image
):

    (
        coarse_left,
        coarse_right,
        y_min,
        y_max
    ) = detect_coarse_sample_edges(
        image
    )

    return refine_sample_edges(
        image,
        coarse_left,
        coarse_right,
        y_min,
        y_max
    )


def x_from_line(
    line,
    y
):

    a, b = line

    return int(
        round(
            a
            * y
            + b
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
        ) / 2.0
    )


# ============================================================
# SCALE / SAMPLE INTERSECTION
# ============================================================

def measurement_line_intersection(
    left_y,
    right_y,
    x_left,
    x_right,
    sample_center_line
):

    dx = (
        x_right
        - x_left
    )

    if abs(dx) < 1e-9:
        return None

    m = (
        right_y
        - left_y
    ) / dx

    c = (
        left_y
        - m
        * x_left
    )

    a, b = (
        sample_center_line
    )

    denominator = (
        1.0
        - m
        * a
    )

    if abs(denominator) < 1e-9:
        return None

    y_cross = (
        m
        * b
        + c
    ) / denominator

    x_cross = (
        a
        * y_cross
        + b
    )

    return (
        float(x_cross),
        float(y_cross)
    )


# ============================================================
# SCALE POSITIONS
# ============================================================

def build_scale_positions(
    measurement_lines,
    left_ruler,
    right_ruler,
    sample_center_line
):

    lx, ly, lw, lh, _ = left_ruler

    rx, ry, rw, rh, _ = right_ruler

    x_start = (
        lx + lw
    )

    x_end = rx

    positions = []

    for line in measurement_lines:

        result = measurement_line_intersection(
            line[
                "left_y"
            ],
            line[
                "right_y"
            ],
            x_start,
            x_end,
            sample_center_line
        )

        if result is None:
            continue

        x_cross, y_cross = result

        positions.append(
            {
                "value":
                    int(
                        line["value"]
                    ),

                "x":
                    float(x_cross),

                "y":
                    float(y_cross),

                "left_y":
                    float(
                        line["left_y"]
                    ),

                "right_y":
                    float(
                        line["right_y"]
                    )
            }
        )

    positions.sort(
        key=lambda item:
        item["value"]
    )

    return positions


# ============================================================
# Y -> HEIGHT
# ============================================================

def front_y_to_height_mm(
    y_front,
    scale_positions
):

    if len(scale_positions) < 2:
        return np.nan

    ordered = sorted(
        scale_positions,
        key=lambda item:
        item["y"]
    )

    ys = np.asarray(
        [
            item["y"]
            for item in ordered
        ],
        dtype=np.float32
    )

    ruler_values = np.asarray(
        [
            item["value"]
            for item in ordered
        ],
        dtype=np.float32
    )

    ruler_value = np.interp(
        float(y_front),
        ys,
        ruler_values
    )

    ruler_value = np.clip(
        ruler_value,
        0.0,
        float(
            TOP_VISIBLE_VALUE
        )
    )

    return float(
        ruler_value
        * MM_PER_RULER_UNIT
    )


# ============================================================
# HEIGHT -> Y
# ============================================================

def height_mm_to_y(
    height_mm,
    scale_positions
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
        dtype=np.float64
    )

    ys = np.asarray(
        [
            item["y"]
            for item in ordered
        ],
        dtype=np.float64
    )

    height_mm = np.clip(
        float(height_mm),
        heights[0],
        heights[-1]
    )

    return float(
        np.interp(
            height_mm,
            heights,
            ys
        )
    )


def find_closest_scale_line(
    y_front,
    scale_positions
):

    if not scale_positions:
        return None

    return min(
        scale_positions,
        key=lambda item:
        abs(
            y_front
            - item["y"]
        )
    )


# ============================================================
# FRONT ANALYSIS REGION
# ============================================================

def prepare_front_geometry(
    image_shape,
    left_sample_line,
    right_sample_line,
    sample_y_min,
    sample_y_max,
    zero_y
):

    h, w = image_shape[:2]

    y_start = max(
        sample_y_min,
        int(
            FRONT_SEARCH_TOP
            * h
        )
    )

    y_end = min(
        sample_y_max,
        h - 1,
        int(
            np.floor(
                zero_y
            )
        )
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
            left_sample_line[0]
            * y
            + left_sample_line[1]
        )

        right = (
            right_sample_line[0]
            * y
            + right_sample_line[1]
        )

        sample_width = (
            right
            - left
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

        x1_values.append(
            x1
        )

        x2_values.append(
            x2
        )

    crop_x0 = max(
        0,
        min(
            x1_values
        ) - 3
    )

    crop_x1 = min(
        w,
        max(
            x2_values
        ) + 4
    )

    crop_width = (
        crop_x1
        - crop_x0
    )

    row_count = (
        y_end
        - y_start
        + 1
    )

    mask = np.zeros(
        (
            row_count,
            crop_width
        ),
        dtype=bool
    )

    for i, (
        x1,
        x2
    ) in enumerate(
        zip(
            x1_values,
            x2_values
        )
    ):

        local_x1 = max(
            0,
            x1
            - crop_x0
        )

        local_x2 = min(
            crop_width,
            x2
            - crop_x0
        )

        if local_x2 > local_x1:

            mask[
                i,
                local_x1:
                local_x2
            ] = True

    return {
        "y_start":
            int(y_start),

        "y_end":
            int(y_end),

        "x_start":
            int(crop_x0),

        "x_end":
            int(crop_x1),

        "mask":
            mask
    }


# ============================================================
# LIQUID FRONT DETECTION
# ============================================================

def extract_front_profile(
    frame,
    front_geometry
):
    """
    Extract the median Lab-L brightness profile through the
    central portion of the porous sample.

    The same geometry and smoothing are used for both the
    reference frame and every analysed frame.
    """

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
        y_start:
        y_end + 1,
        x_start:
        x_end
    ]

    lab = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2LAB
    )

    L = lab[
        :,
        :,
        0
    ].astype(
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

    if np.sum(
        valid
    ) < 20:

        raise RuntimeError(
            "Not enough valid rows for front detection."
        )

    if not np.all(
        valid
    ):

        indices = np.where(
            valid
        )[0]

        profile = np.interp(
            np.arange(
                len(
                    profile
                )
            ),
            indices,
            profile[
                valid
            ]
        )

    image_height = (
        frame.shape[0]
    )

    kernel = odd_number(
        FRONT_SMOOTH_REL
        * image_height
    )

    maximum_kernel = (
        len(
            profile
        )
        if len(
            profile
        ) % 2 == 1
        else len(
            profile
        ) - 1
    )

    kernel = min(
        kernel,
        maximum_kernel
    )

    kernel = max(
        3,
        kernel
    )

    return smooth_1d(
        profile,
        kernel
    )


def estimate_pixels_per_mm(
    scale_positions
):
    """
    Estimate vertical pixel scale from adjacent 1 mm ruler
    positions at the sample centre.
    """

    ordered = sorted(
        scale_positions,
        key=lambda item:
        item[
            "value"
        ]
    )

    differences = []

    for first, second in zip(
        ordered[:-1],
        ordered[1:]
    ):

        value_difference = abs(
            float(
                second[
                    "value"
                ]
            )
            - float(
                first[
                    "value"
                ]
            )
        )

        if value_difference <= 0:
            continue

        pixel_difference = abs(
            float(
                second[
                    "y"
                ]
            )
            - float(
                first[
                    "y"
                ]
            )
        )

        differences.append(
            pixel_difference
            / value_difference
        )

    if not differences:

        raise RuntimeError(
            "Could not estimate pixels per millimetre."
        )

    return float(
        np.median(
            differences
        )
    )


def build_front_reference(
    cap,
    front_geometry,
    frame_count
):
    """
    Build temporal reference information.

    The FIRST video frame remains the actual reference image.
    Additional initial frames are used only to estimate normal
    temporal image noise in the upper part of the sample.

    This is deliberately different from taking the median of the
    first several frames, because capillary rise may start shortly
    after frame 0.
    """

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    profiles = []

    for _ in range(
        max(
            1,
            int(
                frame_count
            )
        )
    ):

        success, frame = cap.read()

        if not success:
            break

        profiles.append(
            extract_front_profile(
                frame,
                front_geometry
            )
        )

    if not profiles:

        raise RuntimeError(
            "Could not build liquid-front reference."
        )

    baseline = np.asarray(
        profiles[0],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Estimate temporal noise using the upper half of the sample.
    #
    # During the first few frames the lower portion is the most
    # likely place for genuine wetting changes. The upper half is
    # therefore a safer place to estimate camera/lighting noise.
    # --------------------------------------------------------

    upper_count = max(
        20,
        int(
            0.50
            * len(
                baseline
            )
        )
    )

    upper_count = min(
        upper_count,
        len(
            baseline
        )
    )

    residual_values = []

    for profile in profiles[1:]:

        profile = np.asarray(
            profile,
            dtype=np.float32
        )

        difference = (
            profile
            - baseline
        )

        # Remove uniform illumination shift.
        shift = float(
            np.median(
                difference[
                    :upper_count
                ]
            )
        )

        residual = (
            difference[
                :upper_count
            ]
            - shift
        )

        residual_values.extend(
            residual.tolist()
        )

    if residual_values:

        residual_values = np.asarray(
            residual_values,
            dtype=np.float32
        )

        centre = float(
            np.median(
                residual_values
            )
        )

        noise = float(
            1.4826
            * np.median(
                np.abs(
                    residual_values
                    - centre
                )
            )
        )

    else:

        noise = 0.0

    # Avoid a zero denominator while not imposing a large
    # artificial threshold.
    noise = max(
        0.25,
        noise
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    return (
        baseline,
        float(
            noise
        )
    )


def evaluate_wetting_candidate(
    candidate_index,
    current_profile,
    baseline_profile,
    pixels_per_mm,
    baseline_noise
):
    """
    Evaluate whether an image edge is actually a wetting front.

    Static edges are present in both the first frame and current
    frame and therefore largely disappear in the temporal
    difference.

    A true wetting front should show:
        dry / less changed material above,
        darker material below.
    """

    candidate_index = int(
        candidate_index
    )

    current_profile = np.asarray(
        current_profile,
        dtype=np.float32
    )

    baseline_profile = np.asarray(
        baseline_profile,
        dtype=np.float32
    )

    change = (
        baseline_profile
        - current_profile
    )

    gap = max(
        1,
        int(
            round(
                WETTING_COMPARE_GAP_MM
                * pixels_per_mm
            )
        )
    )

    window = max(
        4,
        int(
            round(
                WETTING_COMPARE_WINDOW_MM
                * pixels_per_mm
            )
        )
    )

    above_end = max(
        0,
        candidate_index
        - gap
    )

    above_start = max(
        0,
        above_end
        - window
    )

    below_start = min(
        len(
            change
        ),
        candidate_index
        + gap
    )

    below_end = min(
        len(
            change
        ),
        below_start
        + window
    )

    above = change[
        above_start:
        above_end
    ]

    below = change[
        below_start:
        below_end
    ]

    if (
        len(
            above
        ) < 3
        or
        len(
            below
        ) < 3
    ):

        return {
            "valid":
                False,

            "above_change":
                np.nan,

            "below_change":
                np.nan,

            "step":
                np.nan,

            "threshold":
                np.nan,

            "score":
                0.0
        }

    above_change = float(
        np.median(
            above
        )
    )

    below_change = float(
        np.median(
            below
        )
    )

    step = float(
        below_change
        - above_change
    )

    threshold = max(
        float(
            WETTING_MIN_STEP_L
        ),
        float(
            WETTING_NOISE_MULTIPLIER
            * baseline_noise
        )
    )

    minimum_darkening = max(
        float(
            WETTING_MIN_DARKENING_L
        ),
        0.50
        * threshold
    )

    valid = (
        np.isfinite(
            step
        )
        and
        step
        >= threshold
        and
        below_change
        >= minimum_darkening
    )

    score = (
        step
        / max(
            threshold,
            1e-6
        )
    )

    return {
        "valid":
            bool(
                valid
            ),

        "above_change":
            above_change,

        "below_change":
            below_change,

        "step":
            step,

        "threshold":
            threshold,

        "score":
            float(
                score
            )
    }


def detect_liquid_front(
    frame,
    front_geometry,
    baseline_profile,
    baseline_noise,
    pixels_per_mm
):
    """
    Hybrid front detector.

    Stage A:
        Compute the traditional spatial brightness-gradient
        detector. This remains available as a diagnostic raw
        measurement.

    Stage B:
        Instead of trusting only its strongest edge, inspect up
        to FRONT_MAX_CANDIDATES spatial edge candidates.

    Stage C:
        Compare every candidate against frame 0. A candidate is
        accepted as a wetting front only when the material below
        it has actually darkened relative to the material above.

    This is the key change for the beginning of the experiment:
    a static edge near the ruler zero can still have a large
    single-frame gradient and high 'confidence', but it should
    fail the temporal wetting test.
    """

    y_start = int(
        front_geometry[
            "y_start"
        ]
    )

    profile_smooth = extract_front_profile(
        frame,
        front_geometry
    )

    derivative = np.gradient(
        profile_smooth
    )

    edge_strength = (
        -derivative
    )

    background = float(
        np.median(
            edge_strength
        )
    )

    mad = float(
        np.median(
            np.abs(
                edge_strength
                - background
            )
        )
        + 1e-6
    )

    robust_scale = (
        1.4826
        * mad
    )

    # --------------------------------------------------------
    # Traditional strongest-edge result.
    # --------------------------------------------------------

    raw_local_front = int(
        np.argmax(
            edge_strength
        )
    )

    raw_y_front = int(
        y_start
        + raw_local_front
    )

    raw_confidence_score = float(
        (
            edge_strength[
                raw_local_front
            ]
            - background
        )
        /
        robust_scale
    )

    # Preserve temporal evidence for the actual strongest edge,
    # even when it fails the normal wetting threshold. This is
    # used only for retrospective onset review.
    raw_wetting = evaluate_wetting_candidate(
        raw_local_front,
        profile_smooth,
        baseline_profile,
        pixels_per_mm,
        baseline_noise
    )

    raw_wetting_step = float(
        raw_wetting[
            "step"
        ]
    )

    raw_wetting_score = float(
        raw_wetting[
            "score"
        ]
    )

    raw_wetting_valid = bool(
        raw_wetting[
            "valid"
        ]
    )

    # --------------------------------------------------------
    # Generate several possible spatial edges.
    # --------------------------------------------------------

    minimum_peak_strength = (
        background
        + 0.50
        * robust_scale
    )

    minimum_distance = max(
        3,
        int(
            round(
                FRONT_CANDIDATE_MIN_DISTANCE_MM
                * pixels_per_mm
            )
        )
    )

    candidate_indices = find_peaks_1d(
        edge_strength,
        minimum_distance,
        minimum_peak_strength
    )

    if (
        raw_local_front
        not in candidate_indices
    ):

        candidate_indices.append(
            raw_local_front
        )

    candidate_indices = sorted(
        set(
            int(
                value
            )
            for value
            in candidate_indices
        ),
        key=lambda index:
        float(
            edge_strength[
                index
            ]
        ),
        reverse=True
    )

    candidate_indices = (
        candidate_indices[
            :FRONT_MAX_CANDIDATES
        ]
    )

    evaluated = []

    for local_index in candidate_indices:

        edge_score = float(
            (
                edge_strength[
                    local_index
                ]
                - background
            )
            /
            robust_scale
        )

        if (
            edge_score
            < WETTING_MIN_EDGE_SCORE
        ):
            continue

        wetting = evaluate_wetting_candidate(
            local_index,
            profile_smooth,
            baseline_profile,
            pixels_per_mm,
            baseline_noise
        )

        if not wetting[
            "valid"
        ]:
            continue

        # Combined ranking:
        # temporal wetting evidence dominates; spatial edge
        # confidence provides a smaller tie-break contribution.
        combined_score = float(
            wetting[
                "score"
            ]
            + 0.15
            * max(
                0.0,
                edge_score
            )
        )

        evaluated.append(
            {
                "local_index":
                    int(
                        local_index
                    ),

                "global_y":
                    int(
                        y_start
                        + local_index
                    ),

                "edge_score":
                    edge_score,

                "wetting_step":
                    float(
                        wetting[
                            "step"
                        ]
                    ),

                "wetting_score":
                    float(
                        wetting[
                            "score"
                        ]
                    ),

                "combined_score":
                    combined_score
            }
        )

    if evaluated:

        best = max(
            evaluated,
            key=lambda item:
            (
                item[
                    "combined_score"
                ],
                item[
                    "edge_score"
                ]
            )
        )

        validated_y = float(
            best[
                "global_y"
            ]
        )

        validated_confidence_score = float(
            best[
                "edge_score"
            ]
        )

        wetting_step = float(
            best[
                "wetting_step"
            ]
        )

        wetting_score = float(
            best[
                "wetting_score"
            ]
        )

        wetting_detected = True

    else:

        validated_y = np.nan

        validated_confidence_score = 0.0

        wetting_step = 0.0

        wetting_score = 0.0

        wetting_detected = False

    return {
        "raw_y":
            float(
                raw_y_front
            ),

        "raw_confidence_score":
            float(
                raw_confidence_score
            ),

        "raw_wetting_step_L":
            float(
                raw_wetting_step
            ),

        "raw_wetting_score":
            float(
                raw_wetting_score
            ),

        "raw_wetting_valid":
            bool(
                raw_wetting_valid
            ),

        "validated_y":
            float(
                validated_y
            ),

        "validated_confidence_score":
            float(
                validated_confidence_score
            ),

        "wetting_step_L":
            float(
                wetting_step
            ),

        "wetting_score":
            float(
                wetting_score
            ),

        "wetting_detected":
            bool(
                wetting_detected
            ),

        "candidate_count":
            int(
                len(
                    candidate_indices
                )
            )
    }


# ============================================================
# DRAW STATIC GEOMETRY
# ============================================================

def draw_static_geometry(
    image,
    left_ruler,
    right_ruler,
    measurement_lines,
    left_sample_line,
    right_sample_line,
    sample_y_min,
    sample_y_max
):

    out = image.copy()

    h, w = out.shape[:2]

    lx, ly, lw, lh, _ = left_ruler

    rx, ry, rw, rh, _ = right_ruler

    x_start = (
        lx + lw
    )

    x_end = rx

    pad = max(
        5,
        int(
            RULER_BOX_PAD_REL
            * w
        )
    )

    cv2.rectangle(
        out,
        (
            max(
                0,
                lx - pad
            ),
            ly
        ),
        (
            lx + lw,
            ly + lh
        ),
        (255, 255, 0),
        2
    )

    cv2.rectangle(
        out,
        (
            rx,
            ry
        ),
        (
            min(
                w - 1,
                rx + rw + pad
            ),
            ry + rh
        ),
        (255, 255, 0),
        2
    )

    for line in measurement_lines:

        value = line[
            "value"
        ]

        y_left = int(
            round(
                line[
                    "left_y"
                ]
            )
        )

        y_right = int(
            round(
                line[
                    "right_y"
                ]
            )
        )

        colour = (
            255,
            0,
            0
        )

        thickness = 1

        if (
            value <= TOP_MAJOR_VALUE
            and
            value >= BOTTOM_MAJOR_VALUE
            and
            value % 5 == 0
        ):

            colour = (
                255,
                100,
                0
            )

            thickness = 2

        if value == 0:

            colour = (
                0,
                255,
                255
            )

            thickness = 3

        cv2.line(
            out,
            (
                x_start,
                y_left
            ),
            (
                x_end,
                y_right
            ),
            colour,
            thickness,
            cv2.LINE_AA
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
            y0
        ),
        (
            x_from_line(
                left_sample_line,
                y1
            ),
            y1
        ),
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.line(
        out,
        (
            x_from_line(
                right_sample_line,
                y0
            ),
            y0
        ),
        (
            x_from_line(
                right_sample_line,
                y1
            ),
            y1
        ),
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return out


# ============================================================
# DRAW TRACKED FRONT
# ============================================================


def draw_front_overlay(
    image,
    left_sample_line,
    right_sample_line,
    y_front,
    height_mm,
    nearest_line,
    confidence_score,
    time_s,
    measurement_label="measured height"
):

    out = image.copy()

    h, w = out.shape[:2]

    font_scale = max(
        0.65,
        w / 1300.0
    )

    thickness = max(
        2,
        int(
            w / 900
        )
    )

    x_text = int(
        0.02
        * w
    )

    y_text = int(
        0.04
        * h
    )

    step = int(
        0.035
        * h
    )

    cv2.putText(
        out,
        f"time = {time_s:.2f} s",
        (
            x_text,
            y_text
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    if (
        y_front is None
        or
        height_mm is None
        or
        not np.isfinite(
            float(
                y_front
            )
        )
        or
        not np.isfinite(
            float(
                height_mm
            )
        )
    ):

        cv2.putText(
            out,
            "height = unresolved",
            (
                x_text,
                y_text
                + step
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        return out

    y_front = int(
        round(
            float(
                y_front
            )
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
        (
            x_left,
            y_front
        ),
        (
            x_right,
            y_front
        ),
        (0, 0, 255),
        3,
        cv2.LINE_AA
    )

    if nearest_line is not None:

        cv2.circle(
            out,
            (
                int(
                    round(
                        nearest_line[
                            "x"
                        ]
                    )
                ),
                int(
                    round(
                        nearest_line[
                            "y"
                        ]
                    )
                )
            ),
            6,
            (0, 255, 255),
            -1
        )

    cv2.putText(
        out,
        (
            f"{measurement_label} = "
            f"{float(height_mm):.2f} mm"
        ),
        (
            x_text,
            y_text
            + step
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA
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
                y_text
                + 2
                * step
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale
            * 0.8,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    confidence_percent = (
        confidence_score_to_percent(
            confidence_score
        )
    )

    cv2.putText(
        out,
        (
            f"detection confidence = "
            f"{confidence_percent:.1f}%"
        ),
        (
            x_text,
            y_text
            + 3
            * step
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale
        * 0.75,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA
    )

    return out



def save_video_csv(
    output_path,
    results
):

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "frame",
                "time_s",
                "front_y_raw_px",
                "front_y_validated_px",
                "front_y_smoothed_px",
                "front_y_tracking_anchor_px",
                "front_y_final_measured_px",
                "height_raw_mm",
                "height_validated_mm",
                "height_smoothed_mm",
                "height_tracking_anchor_mm",
                "height_final_measured_mm",
                "final_measurement_source",
                "wetting_detected",
                "wetting_step_L",
                "wetting_score",
                "front_candidate_count",
                "tracking_anchor",
                "nearest_ruler_mm",
                "confidence_score_raw",
                "detection_confidence_percent",
                "validated_confidence_score_raw",
                "validated_detection_confidence_percent",
                "raw_wetting_step_L",
                "raw_wetting_score",
                "raw_wetting_valid",
                "tracking_type"
            ]
        )

        for row in results:

            writer.writerow(
                [
                    row[
                        "frame"
                    ],

                    f"{row['time_s']:.6f}",

                    f"{row['front_y_raw']:.3f}",

                    (
                        f"{row['front_y_validated']:.3f}"
                        if np.isfinite(
                            row[
                                "front_y_validated"
                            ]
                        )
                        else ""
                    ),

                    f"{row['front_y_smoothed']:.3f}",

                    (
                        f"{row['front_y_tracked']:.3f}"
                        if np.isfinite(
                            row[
                                "front_y_tracked"
                            ]
                        )
                        else ""
                    ),

                    (
                        f"{row['front_y_final_measured']:.3f}"
                        if np.isfinite(
                            row[
                                "front_y_final_measured"
                            ]
                        )
                        else ""
                    ),

                    f"{row['height_raw_mm']:.6f}",

                    (
                        f"{row['height_validated_mm']:.6f}"
                        if np.isfinite(
                            row[
                                "height_validated_mm"
                            ]
                        )
                        else ""
                    ),

                    f"{row['height_smoothed_mm']:.6f}",

                    (
                        f"{row['height_tracked_mm']:.6f}"
                        if np.isfinite(
                            row[
                                "height_tracked_mm"
                            ]
                        )
                        else ""
                    ),

                    (
                        f"{row['height_final_measured_mm']:.6f}"
                        if np.isfinite(
                            row[
                                "height_final_measured_mm"
                            ]
                        )
                        else ""
                    ),

                    row[
                        "final_measurement_source"
                    ],

                    int(
                        row[
                            "wetting_detected"
                        ]
                    ),

                    f"{row['wetting_step_L']:.6f}",

                    f"{row['wetting_score']:.6f}",

                    row[
                        "candidate_count"
                    ],

                    int(
                        row[
                            "tracking_anchor"
                        ]
                    ),

                    (
                        f"{row['nearest_ruler_mm']:.3f}"
                        if np.isfinite(
                            row[
                                "nearest_ruler_mm"
                            ]
                        )
                        else ""
                    ),

                    f"{row['confidence_score']:.6f}",

                    f"{row['confidence_percent']:.3f}",

                    f"{row['validated_confidence_score']:.6f}",

                    f"{row['validated_confidence_percent']:.3f}",

                    f"{row['raw_wetting_step_L']:.6f}",

                    f"{row['raw_wetting_score']:.6f}",

                    int(
                        row[
                            "raw_wetting_valid"
                        ]
                    ),

                    row[
                        "tracking_type"
                    ]
                ]
            )



def save_individual_plot(
    output_path,
    folder_name,
    video_name,
    results
):

    times = np.asarray(
        [
            row[
                "time_s"
            ]
            for row
            in results
        ],
        dtype=np.float64
    )

    raw = np.asarray(
        [
            row[
                "height_raw_mm"
            ]
            for row
            in results
        ],
        dtype=np.float64
    )

    validated = np.asarray(
        [
            row[
                "height_validated_mm"
            ]
            for row
            in results
        ],
        dtype=np.float64
    )

    final_measured = np.asarray(
        [
            row[
                "height_final_measured_mm"
            ]
            for row
            in results
        ],
        dtype=np.float64
    )

    anchors = np.asarray(
        [
            row[
                "height_tracked_mm"
            ]
            for row
            in results
        ],
        dtype=np.float64
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5
        )
    )

    ax.plot(
        times,
        raw,
        linestyle="-",
        linewidth=0.7,
        alpha=0.25,
        label="Raw strongest edge"
    )

    validated_mask = np.isfinite(
        validated
    )

    ax.scatter(
        times[
            validated_mask
        ],
        validated[
            validated_mask
        ],
        s=5,
        alpha=0.20,
        label="Temporally validated actual measurements"
    )

    final_mask = np.isfinite(
        final_measured
    )

    ax.scatter(
        times[
            final_mask
        ],
        final_measured[
            final_mask
        ],
        s=8,
        alpha=0.85,
        label="Final actual measured series"
    )

    anchor_mask = np.isfinite(
        anchors
    )

    ax.scatter(
        times[
            anchor_mask
        ],
        anchors[
            anchor_mask
        ],
        s=24,
        label="Confirmation / review anchors"
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

    plt.close(
        fig
    )



def save_onset_diagnostic_plot(
    output_path,
    folder_name,
    video_name,
    results
):

    if not results:
        return

    times = np.asarray(
        [
            row[
                "time_s"
            ]
            for row
            in results
        ],
        dtype=np.float64
    )

    raw = np.asarray(
        [
            row[
                "height_raw_mm"
            ]
            for row
            in results
        ],
        dtype=np.float64
    )

    validated = np.asarray(
        [
            row[
                "height_validated_mm"
            ]
            for row
            in results
        ],
        dtype=np.float64
    )

    final_measured = np.asarray(
        [
            row[
                "height_final_measured_mm"
            ]
            for row
            in results
        ],
        dtype=np.float64
    )

    anchors = np.asarray(
        [
            row[
                "height_tracked_mm"
            ]
            for row
            in results
        ],
        dtype=np.float64
    )

    time_mask = (
        times
        <= min(
            2.0,
            float(
                times[
                    -1
                ]
            )
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5
        )
    )

    ax.plot(
        times[
            time_mask
        ],
        raw[
            time_mask
        ],
        linewidth=0.8,
        alpha=0.35,
        label="Strongest spatial edge"
    )

    validated_mask = (
        time_mask
        &
        np.isfinite(
            validated
        )
    )

    ax.scatter(
        times[
            validated_mask
        ],
        validated[
            validated_mask
        ],
        s=16,
        alpha=0.30,
        label="Temporal wetting validation"
    )

    final_mask = (
        time_mask
        &
        np.isfinite(
            final_measured
        )
    )

    ax.scatter(
        times[
            final_mask
        ],
        final_measured[
            final_mask
        ],
        s=26,
        label="Final actual measured series"
    )

    anchor_mask = (
        time_mask
        &
        np.isfinite(
            anchors
        )
    )

    ax.scatter(
        times[
            anchor_mask
        ],
        anchors[
            anchor_mask
        ],
        s=48,
        label="Confirmation / review anchors"
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Capillary rise [mm]"
    )

    ax.set_title(
        f"{folder_name} | {video_name} | Onset diagnostic"
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

    plt.close(
        fig
    )



def calculate_folder_average(
    video_results
):
    """
    Average the final high-resolution ACTUAL measurement series.

    No temporal interpolation is used.

    Runs are aligned by frame index. At each frame, only finite
    actual measurements contribute.
    """

    if len(
        video_results
    ) == 0:
        return None

    lengths = [
        len(
            run[
                "time_s"
            ]
        )
        for run
        in video_results
    ]

    frame_limit = (
        min(
            lengths
        )
        if AVERAGE_USE_COMMON_TIME_ONLY
        else max(
            lengths
        )
    )

    output_time = []

    output_mean = []

    output_std = []

    output_count = []

    for frame_index in range(
        frame_limit
    ):

        values = []

        times_here = []

        for run in video_results:

            if (
                frame_index
                >= len(
                    run[
                        "time_s"
                    ]
                )
            ):
                continue

            value = float(
                run[
                    "height_final_measured_mm"
                ][
                    frame_index
                ]
            )

            if not np.isfinite(
                value
            ):
                continue

            values.append(
                value
            )

            times_here.append(
                float(
                    run[
                        "time_s"
                    ][
                        frame_index
                    ]
                )
            )

        if not values:
            continue

        values = np.asarray(
            values,
            dtype=np.float64
        )

        times_here = np.asarray(
            times_here,
            dtype=np.float64
        )

        output_time.append(
            float(
                np.mean(
                    times_here
                )
            )
        )

        output_mean.append(
            float(
                np.mean(
                    values
                )
            )
        )

        output_std.append(
            float(
                np.std(
                    values,
                    ddof=0
                )
            )
        )

        output_count.append(
            int(
                len(
                    values
                )
            )
        )

    if not output_time:
        raise RuntimeError(
            "No final actual measurements available "
            "for folder averaging."
        )

    return {
        "time_s":
            np.asarray(
                output_time,
                dtype=np.float64
            ),

        "mean_mm":
            np.asarray(
                output_mean,
                dtype=np.float64
            ),

        "std_mm":
            np.asarray(
                output_std,
                dtype=np.float64
            ),

        "n_measurements":
            np.asarray(
                output_count,
                dtype=int
            ),

        "n":
            len(
                video_results
            )
    }



def save_folder_average_csv(
    output_path,
    average
):

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "time_s",
                "mean_final_actual_height_mm",
                "std_final_actual_height_mm",
                "number_of_actual_measurements",
                "number_of_videos_total"
            ]
        )

        for (
            time_s,
            mean,
            std,
            count
        ) in zip(
            average[
                "time_s"
            ],
            average[
                "mean_mm"
            ],
            average[
                "std_mm"
            ],
            average[
                "n_measurements"
            ]
        ):

            writer.writerow(
                [
                    f"{time_s:.6f}",
                    f"{mean:.6f}",
                    f"{std:.6f}",
                    int(
                        count
                    ),
                    int(
                        average[
                            "n"
                        ]
                    )
                ]
            )


def save_folder_average_plot(
    output_path,
    folder_name,
    video_results,
    average
):
    """
    Plot all successfully analysed videos in this experiment
    folder together with the folder average.

    Individual videos:
        - different automatic Matplotlib colours;
        - different marker shapes;
        - thin, semi-transparent lines.

    The average is slightly more prominent but remains
    semi-transparent.

    No missing measurement is interpolated.
    """

    fig, ax = plt.subplots(
        figsize=(
            10,
            6
        )
    )

    marker_cycle = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "P",
        "X",
        "<",
        ">",
        "*",
        "h",
        "p"
    ]

    for video_index, run in enumerate(
        video_results
    ):

        times = np.asarray(
            run[
                "time_s"
            ],
            dtype=np.float64
        )

        values = np.asarray(
            run[
                "height_final_measured_mm"
            ],
            dtype=np.float64
        )

        mask = np.isfinite(
            values
        )

        actual_count = int(
            np.sum(
                mask
            )
        )

        if actual_count == 0:
            continue

        marker_every = max(
            1,
            actual_count
            // 40
        )

        ax.plot(
            times[
                mask
            ],
            values[
                mask
            ],
            linestyle="-",
            linewidth=0.75,
            alpha=0.48,
            marker=marker_cycle[
                video_index
                % len(
                    marker_cycle
                )
            ],
            markersize=3.2,
            markevery=marker_every,
            label=run[
                "video_name"
            ]
        )

    average_time = np.asarray(
        average[
            "time_s"
        ],
        dtype=np.float64
    )

    average_mean = np.asarray(
        average[
            "mean_mm"
        ],
        dtype=np.float64
    )

    average_std = np.asarray(
        average[
            "std_mm"
        ],
        dtype=np.float64
    )

    average_count = np.asarray(
        average[
            "n_measurements"
        ],
        dtype=int
    )

    average_marker_every = max(
        1,
        len(
            average_time
        )
        // 45
    )

    ax.plot(
        average_time,
        average_mean,
        linestyle="-",
        linewidth=1.5,
        alpha=0.70,
        marker=".",
        markersize=4.0,
        markevery=average_marker_every,
        label=(
            f"Average "
            f"(videos={average['n']})"
        )
    )

    spread_mask = (
        average_count
        > 1
    )

    if np.any(
        spread_mask
    ):

        ax.fill_between(
            average_time[
                spread_mask
            ],
            (
                average_mean[
                    spread_mask
                ]
                - average_std[
                    spread_mask
                ]
            ),
            (
                average_mean[
                    spread_mask
                ]
                + average_std[
                    spread_mask
                ]
            ),
            alpha=0.10,
            label="±1 SD"
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

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI
    )

    plt.close(
        fig
    )


def save_folder_comparison_plot(
    output_path,
    folder_name,
    video_results,
    average
):
    """
    Compare all successfully analysed videos in one folder.

    Individual video traces use automatic different colours,
    different markers, and thin semi-transparent lines.
    """

    fig, ax = plt.subplots(
        figsize=(
            10,
            6
        )
    )

    marker_cycle = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "P",
        "X",
        "<",
        ">",
        "*",
        "h",
        "p"
    ]

    for video_index, run in enumerate(
        video_results
    ):

        times = np.asarray(
            run[
                "time_s"
            ],
            dtype=np.float64
        )

        values = np.asarray(
            run[
                "height_final_measured_mm"
            ],
            dtype=np.float64
        )

        mask = np.isfinite(
            values
        )

        actual_count = int(
            np.sum(
                mask
            )
        )

        if actual_count == 0:
            continue

        marker_every = max(
            1,
            actual_count
            // 40
        )

        ax.plot(
            times[
                mask
            ],
            values[
                mask
            ],
            linestyle="-",
            linewidth=0.75,
            alpha=0.48,
            marker=marker_cycle[
                video_index
                % len(
                    marker_cycle
                )
            ],
            markersize=3.2,
            markevery=marker_every,
            label=run[
                "video_name"
            ]
        )

    average_time = np.asarray(
        average[
            "time_s"
        ],
        dtype=np.float64
    )

    average_mean = np.asarray(
        average[
            "mean_mm"
        ],
        dtype=np.float64
    )

    average_marker_every = max(
        1,
        len(
            average_time
        )
        // 45
    )

    ax.plot(
        average_time,
        average_mean,
        linestyle="-",
        linewidth=1.5,
        alpha=0.70,
        marker=".",
        markersize=4.0,
        markevery=average_marker_every,
        label=(
            f"Average "
            f"(videos={average['n']})"
        )
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

    plt.close(
        fig
    )


def save_all_folder_averages_plot(
    output_path,
    root_name,
    folder_averages
):

    if not folder_averages:
        return

    fig, ax = plt.subplots(
        figsize=(
            10,
            6
        )
    )

    for (
        folder_name,
        average
    ) in folder_averages.items():

        ax.scatter(
            average[
                "time_s"
            ],
            average[
                "mean_mm"
            ],
            s=12,
            label=folder_name
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

    plt.close(
        fig
    )


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

    video_path = Path(
        video_path
    )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = video_path.stem

    video_name = video_path.name

    emit_log(
        log_callback,
        f"\n{folder_name} | {video_name}"
    )

    output_video = (
        output_folder
        /
        f"{stem}_marked.mp4"
    )

    output_csv = (
        output_folder
        /
        f"{stem}_data.csv"
    )

    output_plot = (
        output_folder
        /
        f"{stem}_height.png"
    )

    output_geometry = (
        output_folder
        /
        f"{stem}_geometry.png"
    )


    output_onset_diagnostic = (
        output_folder
        /
        f"{stem}_onset_diagnostic.png"
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"{folder_name} | {video_name}"
    )

    print(
        "=" * 70
    )

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open: "
            f"{video_path}"
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
            "Invalid FPS."
        )

    print(
        f"Resolution: "
        f"{width} x {height}"
    )

    print(
        f"FPS: "
        f"{fps:.3f}"
    )

    print(
        f"Frames: "
        f"{total_frames}"
    )

    print(
        f"Duration: "
        f"{total_frames / fps:.2f} s"
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

    # ========================================================
    # STATIC GEOMETRY
    # ========================================================

    emit_log(
        log_callback,
        "Detecting static geometry..."
    )

    success, first_frame = cap.read()

    if not success:

        raise RuntimeError(
            "Could not read first frame."
        )

    (
        geometry_frame,
        geometry_frame_index,
        left_ruler,
        right_ruler
    ) = find_ruler_reference_frame(
        cap,
        first_frame,
        fps,
        total_frames,
        log_callback=log_callback
    )

    left_major_marks = detect_major_ruler_marks(
        geometry_frame,
        left_ruler,
        "left"
    )

    right_major_marks = detect_major_ruler_marks(
        geometry_frame,
        right_ruler,
        "right"
    )

    raw_major_pairs = pair_major_marks(
        left_major_marks,
        right_major_marks,
        height
    )

    major_scale = select_top_down_major_scale(
        raw_major_pairs
    )

    measurement_lines = generate_measurement_lines(
        major_scale
    )

    (
        left_sample_line,
        right_sample_line,
        sample_y_min,
        sample_y_max
    ) = detect_sample_edges(
        geometry_frame
    )

    sample_center_line = get_sample_center_line(
        left_sample_line,
        right_sample_line
    )

    scale_positions = build_scale_positions(
        measurement_lines,
        left_ruler,
        right_ruler,
        sample_center_line
    )

    zero_items = [
        item
        for item in scale_positions
        if item[
            "value"
        ] == 0
    ]

    if len(zero_items) != 1:

        raise RuntimeError(
            "Could not determine calculated zero."
        )

    zero_y = (
        zero_items[0][
            "y"
        ]
    )

    # ========================================================
    # EXTEND SAMPLE GEOMETRY TO THE CALCULATED 0 mm LINE
    # ========================================================
    #
    # sample_y_max is only the lowest row where the side-edge
    # detector found sufficiently strong edge evidence.
    #
    # In some recordings the lower sample edges become weak because
    # of glare, reflection, or the liquid surface. The robust left
    # and right x(y) lines are still valid straight-line fits, so we
    # evaluate those fitted boundaries down to the independently
    # calibrated 0 mm ruler line.
    #
    # This changes spatial ROI geometry only. It does NOT create,
    # interpolate, reconstruct, or synthesize any height-vs-time
    # measurement.
    detected_sample_y_max = int(
        sample_y_max
    )

    sample_analysis_y_max = min(
        height - 1,
        int(
            np.floor(
                zero_y
            )
        )
    )

    if (
        sample_analysis_y_max
        <= sample_y_min
    ):

        raise RuntimeError(
            "Calculated 0 mm line is above the usable sample region."
        )

    emit_log(
        log_callback,
        (
            f"Sample-edge support ends at y={detected_sample_y_max}px; "
            f"analysis geometry extended to 0 mm at "
            f"y={sample_analysis_y_max}px."
        )
    )


    emit_log(
        log_callback,
        (
            f"Static geometry reference frame: "
            f"{geometry_frame_index} "
            f"({geometry_frame_index / fps:.3f} s)."
        )
    )

    front_geometry = prepare_front_geometry(
        first_frame.shape,
        left_sample_line,
        right_sample_line,
        sample_y_min,
        sample_analysis_y_max,
        zero_y
    )

    # ========================================================
    # GEOMETRY CHECK
    # ========================================================

    geometry_overlay = draw_static_geometry(
        geometry_frame,
        left_ruler,
        right_ruler,
        measurement_lines,
        left_sample_line,
        right_sample_line,
        sample_y_min,
        sample_analysis_y_max
    )

    geometry_check = np.hstack(
        (
            geometry_frame,
            geometry_overlay
        )
    )

    cv2.imwrite(
        str(
            output_geometry
        ),
        geometry_check
    )

    # ========================================================
    # PASS 1: RAW + TEMPORALLY VALIDATED FRONT
    # ========================================================

    print()
    print(
        "Pass 1/2: measuring liquid front..."
    )

    emit_log(
        log_callback,
        "Pass 1/2: measuring liquid front..."
    )

    pixels_per_mm = estimate_pixels_per_mm(
        scale_positions
    )

    (
        baseline_profile,
        baseline_noise
    ) = build_front_reference(
        cap,
        front_geometry,
        BASELINE_NOISE_FRAMES
    )

    print(
        f"Temporal wetting reference: "
        f"{pixels_per_mm:.2f} px/mm, "
        f"noise = {baseline_noise:.3f} Lab-L"
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    raw_y = []

    confidence_scores = []

    raw_wetting_steps = []

    raw_wetting_scores = []

    raw_wetting_valid_flags = []

    validated_y = []

    validated_confidence_scores = []

    wetting_steps = []

    wetting_scores = []

    wetting_detected_flags = []

    candidate_counts = []

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        detection = detect_liquid_front(
            frame,
            front_geometry,
            baseline_profile,
            baseline_noise,
            pixels_per_mm
        )

        raw_y.append(
            float(
                detection[
                    "raw_y"
                ]
            )
        )

        confidence_scores.append(
            float(
                detection[
                    "raw_confidence_score"
                ]
            )
        )

        raw_wetting_steps.append(
            float(
                detection[
                    "raw_wetting_step_L"
                ]
            )
        )

        raw_wetting_scores.append(
            float(
                detection[
                    "raw_wetting_score"
                ]
            )
        )

        raw_wetting_valid_flags.append(
            bool(
                detection[
                    "raw_wetting_valid"
                ]
            )
        )

        validated_y.append(
            float(
                detection[
                    "validated_y"
                ]
            )
        )

        validated_confidence_scores.append(
            float(
                detection[
                    "validated_confidence_score"
                ]
            )
        )

        wetting_steps.append(
            float(
                detection[
                    "wetting_step_L"
                ]
            )
        )

        wetting_scores.append(
            float(
                detection[
                    "wetting_score"
                ]
            )
        )

        wetting_detected_flags.append(
            bool(
                detection[
                    "wetting_detected"
                ]
            )
        )

        candidate_counts.append(
            int(
                detection[
                    "candidate_count"
                ]
            )
        )

        frame_number += 1

        if (
            frame_number % 30 == 0
            or
            frame_number == total_frames
        ):

            emit_progress(
                progress_callback,
                stage="measure",
                percent=(
                    100.0
                    * frame_number
                    / max(
                        1,
                        total_frames
                    )
                ),
                frame=frame_number,
                total_frames=total_frames,
                video_name=video_name,
            )

        if (
            frame_number % 100 == 0
            or
            frame_number
            == total_frames
        ):

            percentage = (
                100.0
                * frame_number
                /
                max(
                    1,
                    total_frames
                )
            )

            print(
                f"\rMeasurement: "
                f"{percentage:.1f}%",
                end="",
                flush=True
            )

    print()

    raw_y = np.asarray(
        raw_y,
        dtype=np.float64
    )

    confidence_scores = np.asarray(
        confidence_scores,
        dtype=np.float64
    )

    raw_wetting_steps = np.asarray(
        raw_wetting_steps,
        dtype=np.float64
    )

    raw_wetting_scores = np.asarray(
        raw_wetting_scores,
        dtype=np.float64
    )

    raw_wetting_valid_flags = np.asarray(
        raw_wetting_valid_flags,
        dtype=bool
    )

    validated_y = np.asarray(
        validated_y,
        dtype=np.float64
    )

    validated_confidence_scores = np.asarray(
        validated_confidence_scores,
        dtype=np.float64
    )

    wetting_steps = np.asarray(
        wetting_steps,
        dtype=np.float64
    )

    wetting_scores = np.asarray(
        wetting_scores,
        dtype=np.float64
    )

    wetting_detected_flags = np.asarray(
        wetting_detected_flags,
        dtype=bool
    )

    candidate_counts = np.asarray(
        candidate_counts,
        dtype=int
    )

    actual_frames = len(
        raw_y
    )

    if actual_frames == 0:

        raise RuntimeError(
            "No frames were measured."
        )

    times = (
        np.arange(
            actual_frames,
            dtype=np.float64
        )
        /
        fps
    )

    # ========================================================
    # RAW HEIGHT
    # ========================================================

    height_raw_mm = np.asarray(
        [
            front_y_to_height_mm(
                y,
                scale_positions
            )
            for y in raw_y
        ],
        dtype=np.float64
    )


    # ========================================================
    # TEMPORALLY VALIDATED HEIGHT
    #
    # NaN means that no spatial edge in that frame also showed
    # the expected dry-above / darker-wet-below temporal change.
    # ========================================================

    height_validated_mm = np.asarray(
        [
            (
                front_y_to_height_mm(
                    y,
                    scale_positions
                )
                if np.isfinite(
                    y
                )
                else np.nan
            )
            for y
            in validated_y
        ],
        dtype=np.float64
    )

    # ========================================================
    # DIAGNOSTIC CENTRED MEDIAN
    # ========================================================

    smoothed_y = centered_median_filter(
        raw_y,
        TEMPORAL_WINDOW
    )

    height_smoothed_mm = np.asarray(
        [
            front_y_to_height_mm(
                y,
                scale_positions
            )
            for y in smoothed_y
        ],
        dtype=np.float64
    )

    # ========================================================
    # CONFIDENCE %
    # ========================================================

    confidence_percent = np.asarray(
        [
            confidence_score_to_percent(
                score
            )
            for score
            in confidence_scores
        ],
        dtype=np.float64
    )


    validated_confidence_percent = np.asarray(
        [
            confidence_score_to_percent(
                score
            )
            for score
            in validated_confidence_scores
        ],
        dtype=np.float64
    )

    # Invalid temporal candidates are explicitly removed from
    # tracking participation.
    validated_confidence_percent[
        ~wetting_detected_flags
    ] = 0.0

    print(
        f"Temporally validated front: "
        f"{int(np.sum(wetting_detected_flags))}/"
        f"{actual_frames} frame(s)"
    )

    emit_log(
        log_callback,
        (
            f"Temporally validated front: "
            f"{int(np.sum(wetting_detected_flags))}/"
            f"{actual_frames} frame(s)"
        )
    )

    # ========================================================
    # TRACKED:
    # 3 OF 5 CONFIRMATION
    # ========================================================

    (
        height_tracked_mm,
        tracking_anchor_mask,
        confirmation_records
    ) = build_monotonic_tracked_curve(
        times,
        height_validated_mm,
        validated_confidence_percent,
        raw_height_mm=height_raw_mm,
        raw_confidence_percent=confidence_percent,
        raw_wetting_score=raw_wetting_scores
    )


    # ========================================================
    # FINAL HIGH-RESOLUTION ACTUAL MEASURED SERIES
    # ========================================================

    (
        height_final_measured_mm,
        final_measurement_source
    ) = build_final_measured_series(
        height_validated_mm,
        confirmation_records,
        height_raw_mm
    )

    final_measurement_mask = np.isfinite(
        height_final_measured_mm
    )

    print(
        f"Final actual measured points: "
        f"{int(np.sum(final_measurement_mask))}/"
        f"{actual_frames}"
    )

    emit_log(
        log_callback,
        (
            f"Final actual measured points: "
            f"{int(np.sum(final_measurement_mask))}/"
            f"{actual_frames}"
        )
    )

    print()
    print(
        "Confirmed tracking:"
    )

    print(
        f"  confirmation rule: "
        f"{TRACK_CONFIRM_REQUIRED} of "
        f"{TRACK_CONFIRM_WINDOW}"
    )

    print(
        f"  tolerance: "
        f"+/- {TRACK_CONFIRM_TOLERANCE_MM:.2f} mm"
    )

    print(
        f"  initial tracked height: "
        f"{TRACK_INITIAL_HEIGHT_MM:.2f} mm"
    )

    print(
        f"  first-rise minimum: "
        f"{TRACK_ONSET_MIN_HEIGHT_MM:.2f} mm"
    )

    print(
        f"  confirmed anchors: "
        f"{np.sum(tracking_anchor_mask)}"
    )

    print()

    reviewed_count = sum(
        1
        for record
        in confirmation_records
        if record.get(
            "type"
        ) == "reviewed"
    )

    onset_reviewed_count = sum(
        1
        for record
        in confirmation_records
        if record.get(
            "type"
        ) == "onset_reviewed"
    )

    strong_count = sum(
        1
        for record
        in confirmation_records
        if record.get(
            "type"
        ) == "confirmed"
    )

    print(
        f"  strong 3-of-5 anchors: "
        f"{strong_count}"
    )

    print(
        f"  retrospectively reviewed anchors: "
        f"{reviewed_count}"
    )

    print(
        f"  onset-backtracked actual measurements: "
        f"{onset_reviewed_count}"
    )

    print(
        "  temporal interpolation: DISABLED"
    )

    emit_log(
        log_callback,
        (
            f"Tracking anchors: "
            f"{int(np.sum(tracking_anchor_mask))} | "
            f"Strong: {strong_count} | "
            f"Reviewed: {reviewed_count} | "
            f"Onset backtracked: {onset_reviewed_count} | "
            f"Temporal interpolation: disabled"
        )
    )

    print()

    # Print the beginning in detail because that is where onset
    # problems are most important. Avoid dumping hundreds of
    # repetitive later anchors to the terminal.
    records_to_print = (
        confirmation_records[
            :min(
                30,
                len(
                    confirmation_records
                )
            )
        ]
    )

    for record in records_to_print:

        if record[
            "anchor_index"
        ] == 0:

            print(
                f"  frame 0: "
                f"{record['height']:.3f} mm "
                f"(initial)"
            )

            continue

        support_text = ", ".join(
            str(
                int(i)
            )
            for i
            in record[
                "support_indices"
            ]
        )

        record_type = record.get(
            "type",
            "confirmed"
        )

        if record_type == "onset_reviewed":

            print(
                f"  [{record_type}] "
                f"frame "
                f"{record['anchor_index']}: "
                f"{record['height']:.3f} mm "
                f"(actual raw frame measurement)"
            )

        else:

            rule_window = (
                TRACK_REVIEW_WINDOW
                if record_type
                == "reviewed"
                else TRACK_CONFIRM_WINDOW
            )

            print(
                f"  [{record_type}] "
                f"frame "
                f"{record['anchor_index']}: "
                f"{record['height']:.3f} mm "
                f"supported by "
                f"{record['support_count']}/"
                f"{rule_window} "
                f"frames [{support_text}]"
            )

    if (
        len(
            confirmation_records
        )
        > len(
            records_to_print
        )
    ):

        print(
            f"  ... "
            f"{len(confirmation_records) - len(records_to_print)} "
            f"later anchor record(s) not printed"
        )

    # ========================================================
    # TRACKED HEIGHT -> PIXEL Y
    # ========================================================

    tracked_y = np.asarray(
        [
            (
                height_mm_to_y(
                    value,
                    scale_positions
                )
                if np.isfinite(
                    value
                )
                else np.nan
            )
            for value
            in height_tracked_mm
        ],
        dtype=np.float64
    )


    final_measured_y = np.asarray(
        [
            (
                height_mm_to_y(
                    value,
                    scale_positions
                )
                if np.isfinite(
                    value
                )
                else np.nan
            )
            for value
            in height_final_measured_mm
        ],
        dtype=np.float64
    )

    # ========================================================
    # NEAREST RULER
    # ========================================================

    nearest_ruler_mm = []

    for y in final_measured_y:

        if not np.isfinite(
            y
        ):

            nearest_ruler_mm.append(
                np.nan
            )

            continue

        nearest = find_closest_scale_line(
            y,
            scale_positions
        )

        if nearest is None:

            nearest_ruler_mm.append(
                np.nan
            )

        else:

            nearest_ruler_mm.append(
                float(
                    nearest[
                        "value"
                    ]
                    * MM_PER_RULER_UNIT
                )
            )

    nearest_ruler_mm = np.asarray(
        nearest_ruler_mm,
        dtype=np.float64
    )

    # ========================================================
    # RESULTS
    # ========================================================

    tracking_type_by_frame = {
        int(
            record[
                "anchor_index"
            ]
        ):
        str(
            record.get(
                "type",
                ""
            )
        )

        for record
        in confirmation_records
    }

    results = []

    for i in range(
        actual_frames
    ):

        results.append(
            {
                "frame":
                    int(i),

                "time_s":
                    float(
                        times[i]
                    ),

                "front_y_raw":
                    float(
                        raw_y[i]
                    ),

                "front_y_smoothed":
                    float(
                        smoothed_y[i]
                    ),

                "front_y_tracked":
                    float(
                        tracked_y[i]
                    ),

                "height_raw_mm":
                    float(
                        height_raw_mm[i]
                    ),

                "front_y_validated":
                    float(
                        validated_y[i]
                    ),

                "height_validated_mm":
                    float(
                        height_validated_mm[i]
                    ),

                "wetting_detected":
                    bool(
                        wetting_detected_flags[i]
                    ),

                "wetting_step_L":
                    float(
                        wetting_steps[i]
                    ),

                "wetting_score":
                    float(
                        wetting_scores[i]
                    ),

                "candidate_count":
                    int(
                        candidate_counts[i]
                    ),

                "validated_confidence_score":
                    float(
                        validated_confidence_scores[i]
                    ),

                "validated_confidence_percent":
                    float(
                        validated_confidence_percent[i]
                    ),

                "height_smoothed_mm":
                    float(
                        height_smoothed_mm[i]
                    ),

                "height_tracked_mm":
                    float(
                        height_tracked_mm[i]
                    ),

                "height_final_measured_mm":
                    float(
                        height_final_measured_mm[i]
                    ),

                "front_y_final_measured":
                    float(
                        final_measured_y[i]
                    ),

                "final_measurement_source":
                    str(
                        final_measurement_source[i]
                    ),

                "tracking_anchor":
                    bool(
                        tracking_anchor_mask[i]
                    ),

                "nearest_ruler_mm":
                    float(
                        nearest_ruler_mm[i]
                    ),

                "confidence_score":
                    float(
                        confidence_scores[i]
                    ),

                "confidence_percent":
                    float(
                        confidence_percent[i]
                    ),

                "raw_wetting_step_L":
                    float(
                        raw_wetting_steps[i]
                    ),

                "raw_wetting_score":
                    float(
                        raw_wetting_scores[i]
                    ),

                "raw_wetting_valid":
                    bool(
                        raw_wetting_valid_flags[i]
                    ),

                "tracking_type":
                    tracking_type_by_frame.get(
                        int(
                            i
                        ),
                        ""
                    )
            }
        )

    # ========================================================
    # PASS 2: MARKED VIDEO
    # ========================================================

    if WRITE_MARKED_VIDEO:

        print()
        print(
            "Pass 2/2: writing marked video..."
        )

        emit_log(
            log_callback,
            "Pass 2/2: writing marked video..."
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        fourcc = cv2.VideoWriter_fourcc(
            *VIDEO_CODEC
        )

        writer = cv2.VideoWriter(
            str(
                output_video
            ),
            fourcc,
            fps,
            (
                width,
                height
            )
        )

        if not writer.isOpened():

            raise RuntimeError(
                "Could not create marked video."
            )

        frame_number = 0

        while (
            frame_number
            < actual_frames
        ):

            success, frame = cap.read()

            if not success:
                break

            final_value = float(
                height_final_measured_mm[
                    frame_number
                ]
            )

            final_source = str(
                final_measurement_source[
                    frame_number
                ]
            )

            if np.isfinite(
                final_value
            ):

                y_display = float(
                    final_measured_y[
                        frame_number
                    ]
                )

                display_height = final_value

                if (
                    final_source
                    == "onset_reviewed_raw"
                ):

                    display_confidence = float(
                        confidence_scores[
                            frame_number
                        ]
                    )

                    measurement_label = (
                        "retrospectively accepted actual height"
                    )

                elif (
                    final_source
                    == "initial_condition"
                ):

                    display_confidence = 0.0

                    measurement_label = (
                        "initial height"
                    )

                else:

                    display_confidence = float(
                        validated_confidence_scores[
                            frame_number
                        ]
                    )

                    measurement_label = (
                        "actual measured height"
                    )

            else:

                y_display = None

                display_height = None

                display_confidence = 0.0

                measurement_label = (
                    "height"
                )

            if (
                y_display is not None
                and
                np.isfinite(
                    y_display
                )
            ):

                nearest = find_closest_scale_line(
                    y_display,
                    scale_positions
                )

            else:

                nearest = None

            static_frame = draw_static_geometry(
                frame,
                left_ruler,
                right_ruler,
                measurement_lines,
                left_sample_line,
                right_sample_line,
                sample_y_min,
                sample_analysis_y_max
            )

            marked = draw_front_overlay(
                static_frame,
                left_sample_line,
                right_sample_line,
                y_display,
                display_height,
                nearest,
                display_confidence,
                times[
                    frame_number
                ],
                measurement_label=measurement_label
            )

            writer.write(
                marked
            )

            frame_number += 1

            if (
                frame_number % 30 == 0
                or
                frame_number == actual_frames
            ):

                emit_progress(
                    progress_callback,
                    stage="write",
                    percent=(
                        100.0
                        * frame_number
                        / max(
                            1,
                            actual_frames
                        )
                    ),
                    frame=frame_number,
                    total_frames=actual_frames,
                    video_name=video_name,
                )

            if (
                frame_number % 100 == 0
                or
                frame_number
                == actual_frames
            ):

                percentage = (
                    100.0
                    * frame_number
                    /
                    actual_frames
                )

                print(
                    f"\rVideo writing: "
                    f"{percentage:.1f}%",
                    end="",
                    flush=True
                )

        print()

        writer.release()

    cap.release()

    # ========================================================
    # SAVE
    # ========================================================

    save_video_csv(
        output_csv,
        results
    )

    save_individual_plot(
        output_plot,
        folder_name,
        video_name,
        results
    )


    save_onset_diagnostic_plot(
        output_onset_diagnostic,
        folder_name,
        video_name,
        results
    )

    print()
    print(
        f"Saved CSV: "
        f"{output_csv.name}"
    )

    print(
        f"Saved plot: "
        f"{output_plot.name}"
    )

    if WRITE_MARKED_VIDEO:

        print(
            f"Saved marked video: "
            f"{output_video.name}"
        )

    emit_log(
        log_callback,
        f"Saved outputs for {video_name}"
    )

    return {
        "video_name":
            video_name,

        "video_stem":
            stem,

        "time_s":
            times,

        "height_raw_mm":
            height_raw_mm,

        "height_validated_mm":
            height_validated_mm,

        "height_smoothed_mm":
            height_smoothed_mm,

        # Confirmation/review anchors only.
        "height_tracked_mm":
            height_tracked_mm,

        # High-resolution ACTUAL measured series used for
        # averaging. Unresolved frames remain NaN.
        "height_final_measured_mm":
            height_final_measured_mm
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


def should_ignore_video(
    file_path
):
    """
    Ignore any MP4 whose filename contains 'ignore' anywhere,
    case-insensitively.

    Examples:
        1001_TOP_IGNORE.mp4
        1243r_ignore.mp4
        ignore_2.mp4
        dsc1ignore2123.mp4
    """

    return (
        "ignore"
        in Path(
            file_path
        ).name.lower()
    )


def find_videos_in_folder(
    folder
):
    """
    Return only MP4 files that are allowed to participate in the
    analysis.

    Files containing 'ignore' anywhere in their filename are not
    counted, processed, averaged, or plotted.
    """

    return sorted(
        [
            file
            for file in Path(
                folder
            ).iterdir()

            if (
                file.is_file()
                and
                file.suffix.lower()
                == ".mp4"
                and
                not should_ignore_video(
                    file
                )
            )
        ],
        key=lambda p:
        p.name.lower()
    )


def find_ignored_videos_in_folder(
    folder
):
    """
    Return ignored MP4 files for GUI/log reporting only.
    """

    return sorted(
        [
            file
            for file in Path(
                folder
            ).iterdir()

            if (
                file.is_file()
                and
                file.suffix.lower()
                == ".mp4"
                and
                should_ignore_video(
                    file
                )
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

    ignored_videos = (
        find_ignored_videos_in_folder(
            folder_path
        )
    )

    if ignored_videos:

        emit_log(
            log_callback,
            (
                f"Ignored {len(ignored_videos)} MP4 file(s) "
                f"in {folder_name} because the filename "
                f"contains 'ignore':"
            )
        )

        for ignored_video in ignored_videos:

            emit_log(
                log_callback,
                f"  - {ignored_video.name}"
            )

    if not videos:
        emit_log(
            log_callback,
            (
                f"No analyzable MP4 files in "
                f"{folder_name}; skipped."
            )
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
        video_results,
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
            f"Saved actual-measurement average and comparison "
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

    ignored_total = sum(
        len(
            find_ignored_videos_in_folder(
                folder
            )
        )
        for folder
        in experiment_folders
    )

    emit_log(
        log_callback,
        f"Total MP4 videos to analyze: {total_videos}"
    )

    if ignored_total > 0:

        emit_log(
            log_callback,
            (
                f"Ignored MP4 videos: "
                f"{ignored_total}"
            )
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
