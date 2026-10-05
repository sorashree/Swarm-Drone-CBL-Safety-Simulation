"""
Global configuration for the 8-UAV distributed formation simulation.
"""

import numpy as np


# ============================================================
# SWARM
# ============================================================

NUM_DRONES = 8

# Radius of the octagonal formation
FORMATION_RADIUS = 2.5

# Minimum permitted distance between drones
SAFETY_DISTANCE = 1.20


# ============================================================
# SIMULATION
# ============================================================

DT = 0.02
SIM_TIME = 36.0

SEED = 2025


# ============================================================
# FORMATION CONTROLLER
# ============================================================

KP = 2.2
KV = 1.8

# Strength of graph-based distributed coordination
K_GRAPH = 1.2


# ============================================================
# CONTROL BARRIER FUNCTION
# ============================================================

CBF_ALPHA1 = 3.0
CBF_ALPHA0 = 4.0

# Small numerical safety margin
CBF_SAFETY_MARGIN = 0.02


# ============================================================
# POINT-MASS LIMITS
# ============================================================

MAX_ACCELERATION = 8.0
MAX_SPEED = 5.0


# ============================================================
# VIRTUAL LEADER
# ============================================================

LEADER_RADIUS = 3.0

LEADER_OMEGA = 0.18

LEADER_CENTER = np.array(
    [0.0, 0.0],
    dtype=float
)


# ============================================================
# INITIAL CONDITIONS
# ============================================================

INITIAL_POSITION_NOISE = 0.70
INITIAL_VELOCITY_NOISE = 0.25


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_DIR = "outputs"

ANIMATION_FILE = (
    "outputs/swarm_animation.mp4"
)

RESULT_FILE = (
    "outputs/swarm_result.png"
)
