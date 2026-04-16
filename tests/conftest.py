import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the project root is on sys.path so tests can import project modules.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def landmarks_478():
    """Synthetic 478x3 pixel-coordinate landmark array for a 960x540 frame.

    Key landmark indices are placed at approximate anatomical positions so that
    EAR, MAR, gaze, and head-pose functions return plausible values.  All other
    points are scattered around the face center with a fixed random seed for
    reproducibility.
    """
    pts = np.zeros((478, 3), dtype=np.float64)
    cx, cy = 480.0, 270.0

    rng = np.random.default_rng(42)
    pts[:, 0] = cx + rng.uniform(-80, 80, 478)
    pts[:, 1] = cy + rng.uniform(-100, 100, 478)

    # Right eye: EAR indices [33, 160, 158, 133, 153, 144]
    pts[33]  = [420, 240, 0]   # outer corner
    pts[133] = [460, 240, 0]   # inner corner
    pts[160] = [430, 232, 0]   # upper-outer
    pts[158] = [450, 232, 0]   # upper-inner
    pts[153] = [450, 248, 0]   # lower-inner
    pts[144] = [430, 248, 0]   # lower-outer

    # Left eye: EAR indices [263, 387, 385, 362, 380, 373]
    pts[263] = [500, 240, 0]
    pts[362] = [540, 240, 0]
    pts[387] = [510, 232, 0]
    pts[385] = [530, 232, 0]
    pts[380] = [530, 248, 0]
    pts[373] = [510, 248, 0]

    # Mouth corners and vertical pairs
    pts[78]  = [450, 310, 0]
    pts[308] = [510, 310, 0]
    pts[13]  = [480, 305, 0]
    pts[14]  = [480, 315, 0]
    pts[82]  = [465, 306, 0]
    pts[87]  = [465, 314, 0]
    pts[312] = [495, 306, 0]
    pts[317] = [495, 314, 0]

    # Iris landmarks: RIGHT_IRIS [468-472], LEFT_IRIS [473-477]
    for i in range(468, 473):
        pts[i] = [440, 240, 0]
    for i in range(473, 478):
        pts[i] = [520, 240, 0]

    # Eye lid landmarks for gaze ratio
    pts[159] = [440, 234, 0]  # right upper lid
    pts[145] = [440, 246, 0]  # right lower lid
    pts[386] = [520, 234, 0]  # left upper lid
    pts[374] = [520, 246, 0]  # left lower lid

    # Head-pose indices: [1, 33, 61, 199, 263, 291]
    pts[1]   = [480, 260, 0]
    pts[61]  = [450, 280, 0]
    pts[199] = [480, 340, 0]
    pts[291] = [510, 280, 0]

    return pts
