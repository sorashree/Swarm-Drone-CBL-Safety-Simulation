"""
Formation geometry and virtual-leader trajectory.

The virtual leader follows a figure-eight (∞ / lemniscate)
trajectory while the drones maintain a regular 8-gon formation
around the moving leader.
"""

import numpy as np
import config


def leader_state(t: float):
    """
    Figure-eight (∞) virtual-leader trajectory.

    The trajectory is:

        x(t) = A sin(theta)
        y(t) = B sin(theta) cos(theta)

    where:

        theta = omega * t

    Returns
    -------
    position : np.ndarray
        Virtual leader position [x, y].

    velocity : np.ndarray
        Virtual leader velocity [vx, vy].

    acceleration : np.ndarray
        Virtual leader acceleration [ax, ay].
    """

    # ---------------------------------------------------------
    # Figure-eight trajectory parameters
    # ---------------------------------------------------------

    A = 3.5          # horizontal size of infinity
    B = 2.2          # vertical size of infinity
    omega = 0.35     # angular speed

    theta = omega * t

    # ---------------------------------------------------------
    # Position
    # ---------------------------------------------------------

    x = A * np.sin(theta)

    y = B * np.sin(theta) * np.cos(theta)

    position = np.array([
        x,
        y
    ])

    # ---------------------------------------------------------
    # Velocity
    # ---------------------------------------------------------

    vx = A * omega * np.cos(theta)

    vy = B * omega * np.cos(2.0 * theta)

    velocity = np.array([
        vx,
        vy
    ])

    # ---------------------------------------------------------
    # Acceleration
    # ---------------------------------------------------------

    ax = -A * omega**2 * np.sin(theta)

    ay = -2.0 * B * omega**2 * np.sin(2.0 * theta)

    acceleration = np.array([
        ax,
        ay
    ])

    return (
        position,
        velocity,
        acceleration
    )


def formation_offsets(
    num_drones=config.NUM_DRONES,
    radius=config.FORMATION_RADIUS
):
    """
    Create evenly spaced points around a circle.

    For 8 drones this becomes a regular octagon.

    Drone positions are defined relative to the
    virtual leader.
    """

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        num_drones,
        endpoint=False
    )

    offsets = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])

    return offsets


def desired_formation(t: float):
    """
    Calculate desired position and velocity
    for every drone.

    Desired position:

        p_i^d(t) = p_leader(t) + r_i

    where:

        p_leader(t) = virtual leader position
        r_i         = fixed formation offset

    All drones share the leader velocity because
    the formation translates with the leader.
    """

    (
        center,
        leader_velocity,
        leader_acceleration
    ) = leader_state(t)

    # ---------------------------------------------------------
    # Fixed octagonal formation around leader
    # ---------------------------------------------------------

    offsets = formation_offsets()

    # ---------------------------------------------------------
    # Desired drone positions
    # ---------------------------------------------------------

    desired_positions = (
        center + offsets
    )

    # ---------------------------------------------------------
    # Desired drone velocities
    # ---------------------------------------------------------

    desired_velocities = np.tile(
        leader_velocity,
        (config.NUM_DRONES, 1)
    )

    return (
        desired_positions,
        desired_velocities,
        leader_acceleration,
        center,
        offsets
    )