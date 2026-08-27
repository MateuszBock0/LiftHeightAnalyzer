import cv2
import numpy as np
import csv
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent

ROOT_FOLDER = (
    SCRIPT_FOLDER
    / "Experiments"
)

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


# True:
# interpolate continuously between confirmed anchors.
#
# IMPORTANT:
# The first interpolation no longer starts at frame 0. The curve
# stays exactly at zero until the first confirmed rise begins.
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
# CONFIDENCE DISPLAY
# ============================================================

CONFIDENCE_PERCENT_SCALE = 5.0


# ============================================================
# PLOTTING
# ============================================================

FIGURE_DPI = 200

AVERAGE_USE_COMMON_TIME_ONLY = True


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
    Find a new level supported by at least 3 of 5 frames.

    A frame participates only if:

        confidence >= TRACK_MIN_CONFIDENCE_PERCENT

    A group supports the same level if its measurements lie
    within +/- TRACK_CONFIRM_TOLERANCE_MM of the candidate.

    Among all valid clusters we select the HIGHEST confirmed
    cluster, because this is an upward capillary-rise tracker.

    Returns
    -------
    None

    OR

    {
        "height": confirmed median height,
        "anchor_index": representative frame index,
        "support_indices": indices of supporting frames,
        "support_count": number of agreeing frames
    }
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
        np.isfinite(heights)
        &
        np.isfinite(confidences)
        &
        (
            confidences
            >= TRACK_MIN_CONFIDENCE_PERCENT
        )
    )

    if np.sum(valid) < TRACK_CONFIRM_REQUIRED:

        return None

    valid_heights = heights[
        valid
    ]

    valid_indices = global_indices[
        valid
    ]

    possible_clusters = []

    # --------------------------------------------------------
    # Try every reliable height as a possible cluster centre.
    # --------------------------------------------------------

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

        # Median prevents one member of the cluster from
        # defining the accepted level.
        confirmed_height = float(
            np.median(
                cluster_heights
            )
        )

        # ----------------------------------------------------
        # INITIAL ZERO-REGION GUARD
        #
        # Before the first real capillary rise has been
        # established, reject persistent edges that remain in
        # the near-zero / liquid-level region.
        #
        # This specifically prevents a stable static edge around
        # e.g. 0.8 mm from becoming the first tracked level.
        # ----------------------------------------------------

        if (
            current_height
            <= TRACK_INITIAL_HEIGHT_MM
            + 1e-9
            and
            confirmed_height
            < TRACK_ONSET_MIN_HEIGHT_MM
        ):

            continue

        # Must actually establish a new record level.
        if (
            confirmed_height
            < current_height
            + TRACK_MIN_RISE_MM
        ):

            continue

        # Representative anchor frame:
        #
        # choose the first supporting frame whose measurement
        # is close to the confirmed cluster median.
        distances = np.abs(
            cluster_heights
            - confirmed_height
        )

        closest_distance = np.min(
            distances
        )

        representative_candidates = (
            cluster_indices[
                distances
                <= closest_distance
                + 1e-9
            ]
        )

        anchor_index = int(
            np.min(
                representative_candidates
            )
        )

        possible_clusters.append(
            {
                "height":
                    confirmed_height,

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

    # --------------------------------------------------------
    # Prefer:
    #
    # 1. highest confirmed level
    # 2. greater support count
    # 3. earlier representative frame
    # --------------------------------------------------------

    best = max(
        possible_clusters,
        key=lambda item:
        (
            item["height"],
            item["support_count"],
            -item["anchor_index"]
        )
    )

    return best



# ============================================================
# RETROSPECTIVE REVIEW OF A LARGE CONFIRMED JUMP
# ============================================================

def find_reviewed_anchors_between(
    heights,
    confidences,
    start_index,
    start_height,
    end_index,
    end_height
):
    """
    Look again inside a large jump between two strong tracking
    states.

    IMPORTANT:
    `heights` passed to this function are NOT the old unrestricted
    raw strongest-edge measurements. They are the temporally
    validated wetting-front candidates produced by the hybrid
    detector.

    Therefore, this review can safely be slightly more permissive
    than the main 3-of-5 rule:

        2 supporting detections inside 7 consecutive frames

    The later strong 3-of-5 anchor acts as an upper bracket.

    Returns a chronological list of weaker, reviewed anchors.
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

    window_size = int(
        TRACK_REVIEW_WINDOW
    )

    required = int(
        TRACK_REVIEW_REQUIRED
    )

    if (
        interval_end
        - interval_start
        < required
    ):

        return []

    candidates = []

    last_window_start = max(
        interval_start,
        interval_end
        - window_size
    )

    for window_start in range(
        interval_start,
        last_window_start + 1
    ):

        window_end = min(
            interval_end,
            window_start
            + window_size
        )

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
                + TRACK_REVIEW_TOLERANCE_MM
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

            reviewed_height = float(
                np.median(
                    cluster_heights
                )
            )

            # The reviewed point must live strictly between the
            # lower and upper strong anchors.
            if (
                reviewed_height
                <= start_height
                + TRACK_REVIEW_MIN_RISE_MM
                or
                reviewed_height
                >= end_height
                - TRACK_REVIEW_MIN_RISE_MM
            ):

                continue

            distances = np.abs(
                cluster_heights
                - reviewed_height
            )

            anchor_index = int(
                np.min(
                    cluster_indices[
                        distances
                        <= np.min(
                            distances
                        )
                        + 1e-9
                    ]
                )
            )

            candidates.append(
                {
                    "anchor_index":
                        anchor_index,

                    "height":
                        reviewed_height,

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

    # --------------------------------------------------------
    # Deduplicate overlapping 7-frame windows.
    #
    # Greedily retain a chronological, monotonically increasing
    # sequence with a meaningful physical height advance.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item:
        (
            item[
                "anchor_index"
            ],
            -item[
                "height"
            ]
        )
    )

    reviewed = []

    current_height = float(
        start_height
    )

    current_index = int(
        start_index
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
                - TRACK_REVIEW_MIN_RISE_MM
            )
        ]

        if not eligible:
            break

        # Prefer the earliest reliable progression point.
        earliest_index = min(
            item[
                "anchor_index"
            ]
            for item
            in eligible
        )

        same_time_group = [
            item
            for item
            in eligible
            if item[
                "anchor_index"
            ] == earliest_index
        ]

        # At the same representative time, use the best-supported
        # and then the highest candidate.
        best = max(
            same_time_group,
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


# ============================================================
# NEW:
# MONOTONIC TRACKER WITH 3-OF-5 CONFIRMATION
# ============================================================

def build_monotonic_tracked_curve(
    time_s,
    raw_height_mm,
    confidence_percent
):
    """
    Monotonic capillary-front reconstruction.

    New levels are NOT accepted from a single high value.

    A new height must be supported by at least:

        3 out of 5 consecutive frames

    within:

        +/- TRACK_CONFIRM_TOLERANCE_MM

    and all supporting frames must satisfy the confidence
    threshold.

    Lower later detections are ignored after a level has been
    confirmed.

    Between confirmed anchors, linear interpolation is used
    when TRACK_INTERPOLATE_BETWEEN_ANCHORS=True.
    """

    time_s = np.asarray(
        time_s,
        dtype=np.float64
    )

    raw_height_mm = np.asarray(
        raw_height_mm,
        dtype=np.float64
    )

    confidence_percent = np.asarray(
        confidence_percent,
        dtype=np.float64
    )

    n = len(
        raw_height_mm
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
        len(time_s) == n
        and
        len(confidence_percent) == n
    ):

        raise ValueError(
            "Tracking arrays must have equal length."
        )

    # --------------------------------------------------------
    # INITIAL LEVEL
    #
    # The physical tracked curve starts at 0 mm.
    #
    # Do NOT use raw_height_mm[0] here. Before the wetting front
    # exists, the raw image detector still has to return the
    # strongest edge somewhere, and that edge can be a static
    # feature close to the liquid level.
    # --------------------------------------------------------

    anchor_indices = [
        0
    ]

    anchor_heights = [
        float(
            TRACK_INITIAL_HEIGHT_MM
        )
    ]

    confirmation_records = [
        {
            "anchor_index":
                0,

            "height":
                float(
                    TRACK_INITIAL_HEIGHT_MM
                ),

            "support_indices":
                np.array(
                    [0],
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

    # --------------------------------------------------------
    # Scan every possible 5-frame consecutive window.
    # --------------------------------------------------------

    window_size = int(
        TRACK_CONFIRM_WINDOW
    )

    required = int(
        TRACK_CONFIRM_REQUIRED
    )

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

    # Last valid complete window starts here.
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

            heights_window = (
                raw_height_mm[
                    start:end
                ]
            )

            confidence_window = (
                confidence_percent[
                    start:end
                ]
            )

            global_indices = np.arange(
                start,
                end,
                dtype=int
            )

            confirmed = (
                find_confirmed_level_in_window(
                    heights_window,
                    confidence_window,
                    global_indices,
                    current_height
                )
            )

            if confirmed is None:
                continue

            confirmed_height = (
                confirmed[
                    "height"
                ]
            )

            anchor_index = (
                confirmed[
                    "anchor_index"
                ]
            )

            # ------------------------------------------------
            # Safety:
            # never add an anchor before or at the previous
            # anchor.
            # ------------------------------------------------

            if (
                anchor_index
                <= anchor_indices[-1]
            ):
                continue

            if (
                confirmed_height
                < current_height
                + TRACK_MIN_RISE_MM
            ):
                continue

            # ------------------------------------------------
            # LARGE-JUMP REVIEW
            #
            # A sudden strong confirmation can mean that the
            # detector intermittently saw the true front earlier
            # but failed the strict 3-of-5 rule.
            #
            # Review that interval using ONLY the temporally
            # validated wetting-front candidates.
            # ------------------------------------------------

            if (
                confirmed_height
                - current_height
                >= TRACK_REVIEW_TRIGGER_MM
            ):

                reviewed_anchors = (
                    find_reviewed_anchors_between(
                        raw_height_mm,
                        confidence_percent,
                        anchor_indices[-1],
                        current_height,
                        anchor_index,
                        confirmed_height
                    )
                )

                for reviewed in reviewed_anchors:

                    reviewed_index = int(
                        reviewed[
                            "anchor_index"
                        ]
                    )

                    reviewed_height = float(
                        reviewed[
                            "height"
                        ]
                    )

                    if (
                        reviewed_index
                        <= anchor_indices[-1]
                    ):
                        continue

                    if (
                        reviewed_height
                        <= current_height
                        + TRACK_REVIEW_MIN_RISE_MM
                    ):
                        continue

                    anchor_indices.append(
                        reviewed_index
                    )

                    anchor_heights.append(
                        reviewed_height
                    )

                    confirmation_records.append(
                        reviewed
                    )

                    current_height = float(
                        reviewed_height
                    )

            # ------------------------------------------------
            # Append the strong 3-of-5 anchor itself.
            # ------------------------------------------------

            if (
                anchor_index
                <= anchor_indices[-1]
            ):
                continue

            if (
                confirmed_height
                < current_height
                + TRACK_MIN_RISE_MM
            ):
                # A reviewed anchor may already explain nearly
                # the whole jump. In that case the strong anchor
                # is still useful as evidence, but there is no
                # need to create a near-duplicate tracking point.
                continue

            anchor_indices.append(
                int(
                    anchor_index
                )
            )

            anchor_heights.append(
                float(
                    confirmed_height
                )
            )

            confirmation_records.append(
                confirmed
            )

            current_height = float(
                confirmed_height
            )

    # --------------------------------------------------------
    # Convert to arrays.
    # --------------------------------------------------------

    anchor_indices = np.asarray(
        anchor_indices,
        dtype=int
    )

    anchor_heights = np.asarray(
        anchor_heights,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Ensure anchors are in chronological order.
    # --------------------------------------------------------

    order = np.argsort(
        anchor_indices
    )

    anchor_indices = (
        anchor_indices[
            order
        ]
    )

    anchor_heights = (
        anchor_heights[
            order
        ]
    )

    # --------------------------------------------------------
    # Remove duplicate anchor indices if any.
    # Keep highest confirmed height at a duplicate time.
    # --------------------------------------------------------

    unique_indices = []

    unique_heights = []

    for index, height in zip(
        anchor_indices,
        anchor_heights
    ):

        if (
            unique_indices
            and
            index
            == unique_indices[-1]
        ):

            unique_heights[-1] = max(
                unique_heights[-1],
                float(
                    height
                )
            )

        else:

            unique_indices.append(
                int(
                    index
                )
            )

            unique_heights.append(
                float(
                    height
                )
            )

    anchor_indices = np.asarray(
        unique_indices,
        dtype=int
    )

    anchor_heights = np.asarray(
        unique_heights,
        dtype=np.float64
    )

    # Physically irreversible:
    anchor_heights = np.maximum.accumulate(
        anchor_heights
    )

    anchor_mask = np.zeros(
        n,
        dtype=bool
    )

    anchor_mask[
        anchor_indices
    ] = True

    # --------------------------------------------------------
    # CONSTRUCT TRACKED CURVE
    #
    # CRITICAL INITIAL-CONDITION RULE:
    #
    # Do NOT interpolate from frame 0 to the first positive
    # confirmed anchor. Doing that invents a gradual capillary
    # rise during frames in which no rise was observed.
    #
    # Instead:
    #   1. hold the curve at exactly 0 mm;
    #   2. locate the first confirmed positive cluster;
    #   3. begin interpolation only around the first supporting
    #      frames of that cluster;
    #   4. continue normal interpolation between later anchors.
    # --------------------------------------------------------

    positive_mask = (
        anchor_heights
        >
        TRACK_INITIAL_HEIGHT_MM
        + 1e-9
    )

    positive_anchor_indices = (
        anchor_indices[
            positive_mask
        ]
    )

    positive_anchor_heights = (
        anchor_heights[
            positive_mask
        ]
    )

    if len(
        positive_anchor_indices
    ) == 0:

        # No real rise was ever confirmed.
        tracked = np.full(
            n,
            float(
                TRACK_INITIAL_HEIGHT_MM
            ),
            dtype=np.float64
        )

    elif (
        TRACK_INTERPOLATE_BETWEEN_ANCHORS
    ):

        first_positive_index = int(
            positive_anchor_indices[0]
        )

        # Default to the frame immediately before the first
        # positive anchor.
        onset_zero_index = max(
            0,
            first_positive_index - 1
        )

        # If possible, use the beginning of the actual 3-of-5
        # support cluster. This limits the reconstructed onset to
        # the short interval in which evidence for the front
        # first appears, instead of spreading it backwards over
        # the whole beginning of the video.
        for record in confirmation_records:

            if (
                int(
                    record[
                        "anchor_index"
                    ]
                )
                == first_positive_index
                and
                float(
                    record[
                        "height"
                    ]
                )
                >
                TRACK_INITIAL_HEIGHT_MM
                + 1e-9
            ):

                support_indices = np.asarray(
                    record[
                        "support_indices"
                    ],
                    dtype=int
                )

                if len(
                    support_indices
                ) > 0:

                    onset_zero_index = max(
                        0,
                        int(
                            np.min(
                                support_indices
                            )
                        )
                        - 1
                    )

                break

        # The zero anchor must remain strictly before the first
        # positive anchor.
        onset_zero_index = min(
            onset_zero_index,
            max(
                0,
                first_positive_index - 1
            )
        )

        interpolation_indices = np.concatenate(
            (
                np.asarray(
                    [
                        onset_zero_index
                    ],
                    dtype=int
                ),
                positive_anchor_indices
            )
        )

        interpolation_heights = np.concatenate(
            (
                np.asarray(
                    [
                        float(
                            TRACK_INITIAL_HEIGHT_MM
                        )
                    ],
                    dtype=np.float64
                ),
                positive_anchor_heights
            )
        )

        # Remove a duplicate index if the first positive anchor
        # somehow occurs immediately at the zero anchor.
        clean_indices = []
        clean_heights = []

        for index, height in zip(
            interpolation_indices,
            interpolation_heights
        ):

            index = int(
                index
            )

            height = float(
                height
            )

            if (
                clean_indices
                and
                index
                == clean_indices[-1]
            ):

                clean_heights[-1] = max(
                    clean_heights[-1],
                    height
                )

            else:

                clean_indices.append(
                    index
                )

                clean_heights.append(
                    height
                )

        interpolation_indices = np.asarray(
            clean_indices,
            dtype=int
        )

        interpolation_heights = np.asarray(
            clean_heights,
            dtype=np.float64
        )

        interpolation_heights = np.maximum.accumulate(
            interpolation_heights
        )

        tracked = np.full(
            n,
            float(
                TRACK_INITIAL_HEIGHT_MM
            ),
            dtype=np.float64
        )

        start_index = int(
            interpolation_indices[0]
        )

        tracked[
            start_index:
        ] = np.interp(
            time_s[
                start_index:
            ],
            time_s[
                interpolation_indices
            ],
            interpolation_heights
        )

    else:

        # Step-wise monotonic tracker:
        # zero until the first confirmed positive anchor, then
        # hold each confirmed level until the next one.
        tracked = np.full(
            n,
            float(
                TRACK_INITIAL_HEIGHT_MM
            ),
            dtype=np.float64
        )

        current = float(
            TRACK_INITIAL_HEIGHT_MM
        )

        anchor_pointer = 0

        for i in range(n):

            while (
                anchor_pointer
                < len(
                    positive_anchor_indices
                )
                and
                i
                >= positive_anchor_indices[
                    anchor_pointer
                ]
            ):

                current = float(
                    positive_anchor_heights[
                        anchor_pointer
                    ]
                )

                anchor_pointer += 1

            tracked[i] = current

    tracked = np.maximum.accumulate(
        tracked
    )

    return (
        tracked,
        anchor_mask,
        confirmation_records
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

    y0 = int(
        0.08 * h
    )

    y1 = int(
        0.97 * h
    )

    band = gray[
        y0:y1,
        :
    ]

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
        (
            1,
            close_width
        ),
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

        active = (
            column_mask[x] > 0
        )

        if (
            active
            and start is None
        ):

            start = x

        if (
            start is not None
            and
            (
                not active
                or x == w - 1
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

            if region_width > 0.04 * w:

                regions.append(
                    (
                        start,
                        end
                    )
                )

            start = None

    left_candidates = []

    right_candidates = []

    for region in regions:

        centre = (
            region[0]
            + region[1]
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
        key=lambda r:
        r[1] - r[0]
    )

    right_region = max(
        right_candidates,
        key=lambda r:
        r[1] - r[0]
    )

    def make_box(region):

        xs, xe = region

        roi = gray[
            :,
            xs:xe + 1
        ]

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

        top = int(
            ys.min()
        )

        bottom = int(
            ys.max()
        )

        width = (
            xe - xs + 1
        )

        height = (
            bottom - top + 1
        )

        return (
            int(xs),
            int(top),
            int(width),
            int(height),
            int(width * height)
        )

    return (
        make_box(left_region),
        make_box(right_region)
    )


# ============================================================
# MAJOR SPACING
# ============================================================

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
        int(
            FRONT_SEARCH_BOTTOM
            * h
        ),
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
    time_s
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

    cv2.putText(
        out,
        (
            f"tracked height = "
            f"{height_mm:.2f} mm"
        ),
        (
            x_text,
            y_text + step
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
                + 2 * step
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.8,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    cv2.putText(
        out,
        (
            f"validated detection confidence = "
            f"{confidence_percent:.1f}%"
        ),
        (
            x_text,
            y_text
            + 3 * step
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.75,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA
    )

    return out


# ============================================================
# SAVE VIDEO CSV
# ============================================================

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
                "front_y_tracked_px",
                "height_raw_mm",
                "height_validated_mm",
                "height_smoothed_mm",
                "height_tracked_mm",
                "wetting_detected",
                "wetting_step_L",
                "wetting_score",
                "front_candidate_count",
                "tracking_anchor",
                "nearest_ruler_mm",
                "confidence_score_raw",
                "detection_confidence_percent",
                "validated_confidence_score_raw",
                "validated_detection_confidence_percent"
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

                    f"{row['front_y_tracked']:.3f}",

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

                    f"{row['height_tracked_mm']:.6f}",

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

                    f"{row['nearest_ruler_mm']:.3f}",

                    f"{row['confidence_score']:.6f}",

                    f"{row['confidence_percent']:.3f}",

                    f"{row['validated_confidence_score']:.6f}",

                    f"{row['validated_confidence_percent']:.3f}"
                ]
            )


# ============================================================
# INDIVIDUAL PLOT
# ============================================================

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
            for row in results
        ]
    )

    raw = np.asarray(
        [
            row[
                "height_raw_mm"
            ]
            for row in results
        ]
    )

    smoothed = np.asarray(
        [
            row[
                "height_smoothed_mm"
            ]
            for row in results
        ]
    )

    validated = np.asarray(
        [
            row[
                "height_validated_mm"
            ]
            for row in results
        ],
        dtype=np.float64
    )

    wetting_mask = np.asarray(
        [
            row[
                "wetting_detected"
            ]
            for row in results
        ],
        dtype=bool
    )

    tracked = np.asarray(
        [
            row[
                "height_tracked_mm"
            ]
            for row in results
        ]
    )

    anchor_mask = np.asarray(
        [
            row[
                "tracking_anchor"
            ]
            for row in results
        ],
        dtype=bool
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
        linewidth=0.8,
        alpha=0.45,
        label="Raw"
    )

    ax.plot(
        times,
        smoothed,
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label="Centred median"
    )

    ax.scatter(
        times[
            wetting_mask
        ],
        validated[
            wetting_mask
        ],
        s=8,
        alpha=0.55,
        label="Temporally validated front"
    )

    ax.plot(
        times,
        tracked,
        linestyle="-",
        linewidth=2.2,
        label="Tracked hybrid front"
    )

    ax.scatter(
        times[
            anchor_mask
        ],
        tracked[
            anchor_mask
        ],
        s=18,
        label="Confirmed tracking anchors"
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



# ============================================================
# ONSET DIAGNOSTIC PLOT
# ============================================================

def save_onset_diagnostic_plot(
    output_path,
    folder_name,
    video_name,
    results
):
    """
    Save an enlarged view of the first two seconds.

    This makes onset behaviour auditable:
        - old strongest spatial edge,
        - temporally validated wetting candidates,
        - final tracked curve.
    """

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

    tracked = np.asarray(
        [
            row[
                "height_tracked_mm"
            ]
            for row
            in results
        ],
        dtype=np.float64
    )

    wetting = np.asarray(
        [
            row[
                "wetting_detected"
            ]
            for row
            in results
        ],
        dtype=bool
    )

    mask = (
        times
        <= min(
            2.0,
            float(
                times[-1]
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
            mask
        ],
        raw[
            mask
        ],
        linewidth=0.8,
        alpha=0.45,
        label="Strongest spatial edge"
    )

    valid_mask = (
        mask
        &
        wetting
        &
        np.isfinite(
            validated
        )
    )

    ax.scatter(
        times[
            valid_mask
        ],
        validated[
            valid_mask
        ],
        s=18,
        label="Temporal wetting validation"
    )

    ax.plot(
        times[
            mask
        ],
        tracked[
            mask
        ],
        linewidth=2.2,
        label="Tracked"
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


# ============================================================
# FOLDER AVERAGE
#
# TRACKED DATA ONLY
# ============================================================

def calculate_folder_average(
    video_results
):

    if len(video_results) == 0:
        return None

    if len(video_results) == 1:

        run = (
            video_results[0]
        )

        return {
            "time_s":
                run[
                    "time_s"
                ].copy(),

            "mean_mm":
                run[
                    "height_tracked_mm"
                ].copy(),

            "std_mm":
                np.zeros_like(
                    run[
                        "height_tracked_mm"
                    ]
                ),

            "n":
                1
        }

    dt_values = []

    for run in video_results:

        times = (
            run[
                "time_s"
            ]
        )

        if len(times) >= 2:

            dt_values.append(
                np.median(
                    np.diff(
                        times
                    )
                )
            )

    if not dt_values:

        raise RuntimeError(
            "Cannot determine average time step."
        )

    dt = float(
        np.max(
            dt_values
        )
    )

    if AVERAGE_USE_COMMON_TIME_ONLY:

        end_time = min(
            float(
                run[
                    "time_s"
                ][-1]
            )
            for run in video_results
        )

    else:

        end_time = max(
            float(
                run[
                    "time_s"
                ][-1]
            )
            for run in video_results
        )

    common_time = np.arange(
        0.0,
        end_time
        + 0.5 * dt,
        dt
    )

    interpolated = []

    for run in video_results:

        curve = np.interp(
            common_time,
            run[
                "time_s"
            ],
            run[
                "height_tracked_mm"
            ]
        )

        interpolated.append(
            curve
        )

    matrix = np.vstack(
        interpolated
    )

    mean = np.mean(
        matrix,
        axis=0
    )

    std = np.std(
        matrix,
        axis=0,
        ddof=0
    )

    return {
        "time_s":
            common_time,

        "mean_mm":
            mean,

        "std_mm":
            std,

        "n":
            len(
                video_results
            )
    }


# ============================================================
# SAVE FOLDER AVERAGE CSV
# ============================================================

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
                "mean_tracked_height_mm",
                "std_tracked_height_mm",
                "number_of_videos"
            ]
        )

        for (
            time_s,
            mean,
            std
        ) in zip(
            average[
                "time_s"
            ],
            average[
                "mean_mm"
            ],
            average[
                "std_mm"
            ]
        ):

            writer.writerow(
                [
                    f"{time_s:.6f}",
                    f"{mean:.6f}",
                    f"{std:.6f}",
                    average[
                        "n"
                    ]
                ]
            )


# ============================================================
# FOLDER AVERAGE PLOT
# ============================================================

def save_folder_average_plot(
    output_path,
    folder_name,
    average
):

    fig, ax = plt.subplots(
        figsize=(
            9,
            5
        )
    )

    time = average[
        "time_s"
    ]

    mean = average[
        "mean_mm"
    ]

    std = average[
        "std_mm"
    ]

    mean_line, = ax.plot(
        time,
        mean,
        linestyle="-",
        linewidth=2.2,
        label="Average tracked rise"
    )

    if average[
        "n"
    ] > 1:

        ax.fill_between(
            time,
            mean - std,
            mean + std,
            color=mean_line.get_color(),
            alpha=0.15,
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
# FOLDER COMPARISON
# ============================================================

def save_folder_comparison_plot(
    output_path,
    folder_name,
    video_results,
    average
):

    fig, ax = plt.subplots(
        figsize=(
            10,
            6
        )
    )

    for run in video_results:

        ax.plot(
            run[
                "time_s"
            ],
            run[
                "height_tracked_mm"
            ],
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label=run[
                "video_name"
            ]
        )

    ax.plot(
        average[
            "time_s"
        ],
        average[
            "mean_mm"
        ],
        linestyle="-",
        linewidth=2.8,
        label=(
            f"Average "
            f"(n={average['n']})"
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


# ============================================================
# ALL FOLDER AVERAGES
# ============================================================

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

        ax.plot(
            average[
                "time_s"
            ],
            average[
                "mean_mm"
            ],
            linewidth=2.0,
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
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    output_folder,
    folder_name
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

    # ========================================================
    # STATIC GEOMETRY
    # ========================================================

    success, first_frame = cap.read()

    if not success:

        raise RuntimeError(
            "Could not read first frame."
        )

    (
        left_ruler,
        right_ruler
    ) = detect_rulers(
        first_frame
    )

    left_major_marks = detect_major_ruler_marks(
        first_frame,
        left_ruler,
        "left"
    )

    right_major_marks = detect_major_ruler_marks(
        first_frame,
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
        first_frame
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

    front_geometry = prepare_front_geometry(
        first_frame.shape,
        left_sample_line,
        right_sample_line,
        sample_y_min,
        sample_y_max,
        zero_y
    )

    # ========================================================
    # GEOMETRY CHECK
    # ========================================================

    geometry_overlay = draw_static_geometry(
        first_frame,
        left_ruler,
        right_ruler,
        measurement_lines,
        left_sample_line,
        right_sample_line,
        sample_y_min,
        sample_y_max
    )

    geometry_check = np.hstack(
        (
            first_frame,
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
        validated_confidence_percent
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
            height_mm_to_y(
                value,
                scale_positions
            )
            for value
            in height_tracked_mm
        ],
        dtype=np.float64
    )

    # ========================================================
    # NEAREST RULER
    # ========================================================

    nearest_ruler_mm = []

    for y in tracked_y:

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

            y_display = int(
                round(
                    tracked_y[
                        frame_number
                    ]
                )
            )

            nearest = find_closest_scale_line(
                tracked_y[
                    frame_number
                ],
                scale_positions
            )

            static_frame = draw_static_geometry(
                frame,
                left_ruler,
                right_ruler,
                measurement_lines,
                left_sample_line,
                right_sample_line,
                sample_y_min,
                sample_y_max
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
                validated_confidence_scores[
                    frame_number
                ],
                times[
                    frame_number
                ]
            )

            writer.write(
                marked
            )

            frame_number += 1

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

        # THIS is used for folder averaging.
        "height_tracked_mm":
            height_tracked_mm
    }


# ============================================================
# PROCESS ONE EXPERIMENT FOLDER
# ============================================================

def process_experiment_folder(
    folder_path,
    output_root
):

    folder_path = Path(
        folder_path
    )

    folder_name = (
        folder_path.name
    )

    output_folder = (
        output_root
        /
        folder_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    videos = sorted(
        [
            file
            for file
            in folder_path.iterdir()

            if (
                file.is_file()
                and
                file.suffix.lower()
                == ".mp4"
            )
        ]
    )

    if not videos:

        print(
            f"No MP4 files in "
            f"{folder_name}"
        )

        return None

    print()
    print(
        "#" * 70
    )

    print(
        f"FOLDER: "
        f"{folder_name}"
    )

    print(
        f"Videos: "
        f"{len(videos)}"
    )

    print(
        "#" * 70
    )

    video_results = []

    for video_path in videos:

        try:

            result = process_video(
                video_path,
                output_folder,
                folder_name
            )

            video_results.append(
                result
            )

        except Exception as error:

            print()
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

            print(
                f"FAILED: "
                f"{folder_name} | "
                f"{video_path.name}"
            )

            print(
                f"Reason: "
                f"{error}"
            )

            print(
                "Continuing..."
            )

            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

    if not video_results:

        return None

    average = calculate_folder_average(
        video_results
    )

    average_csv = (
        output_folder
        /
        f"{folder_name}_average.csv"
    )

    average_plot = (
        output_folder
        /
        f"{folder_name}_average.png"
    )

    comparison_plot = (
        output_folder
        /
        f"{folder_name}_comparison.png"
    )

    save_folder_average_csv(
        average_csv,
        average
    )

    save_folder_average_plot(
        average_plot,
        folder_name,
        average
    )

    save_folder_comparison_plot(
        comparison_plot,
        folder_name,
        video_results,
        average
    )

    return {
        "folder_name":
            folder_name,

        "average":
            average,

        "video_results":
            video_results
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not ROOT_FOLDER.exists():

        raise FileNotFoundError(
            f"Experiments folder does not exist:\n"
            f"{ROOT_FOLDER}"
        )

    if not ROOT_FOLDER.is_dir():

        raise RuntimeError(
            f"ROOT_FOLDER is not a directory:\n"
            f"{ROOT_FOLDER}"
        )

    output_root = (
        ROOT_FOLDER
        /
        OUTPUT_ROOT_NAME
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    experiment_folders = sorted(
        [
            folder
            for folder
            in ROOT_FOLDER.iterdir()

            if (
                folder.is_dir()
                and
                folder.name
                != OUTPUT_ROOT_NAME
            )
        ]
    )

    if not experiment_folders:

        raise RuntimeError(
            "No experimental subfolders found."
        )

    print()
    print(
        "=" * 70
    )

    print(
        "CAPILLARY-RISE BATCH ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        ROOT_FOLDER
    )

    folder_results = []

    for folder in experiment_folders:

        try:

            result = process_experiment_folder(
                folder,
                output_root
            )

            if result is not None:

                folder_results.append(
                    result
                )

        except Exception as error:

            print()
            print(
                f"FAILED FOLDER: "
                f"{folder.name}"
            )

            print(
                f"Reason: "
                f"{error}"
            )

    # ========================================================
    # FINAL COMPARISON OF TRACKED AVERAGES
    # ========================================================

    folder_averages = {
        result[
            "folder_name"
        ]:
        result[
            "average"
        ]

        for result
        in folder_results
    }

    if folder_averages:

        final_plot = (
            output_root
            /
            "all_folder_averages.png"
        )

        save_all_folder_averages_plot(
            final_plot,
            ROOT_FOLDER.name,
            folder_averages
        )

    print()
    print(
        "=" * 70
    )

    print(
        "FINISHED"
    )

    print(
        "=" * 70
    )

    print(
        output_root
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
