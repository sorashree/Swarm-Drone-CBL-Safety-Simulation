"""
Distributed formation-control law.
"""

import numpy as np

import config

from graph import distributed_error


def nominal_control(
    positions,
    velocities,
    desired_positions,
    desired_velocities,
    leader_acceleration,
    L
):

    """
    Distributed formation controller.

    e_i = p_i - p_i^d

    e_dot_i = v_i - v_i^d

    u_nom =
        a_c
        - Kp * e
        - Kv * e_dot
        - Kgraph * L*e
    """

    # --------------------------------
    # Position error
    # --------------------------------

    position_error = (
        positions -
        desired_positions
    )


    # --------------------------------
    # Velocity error
    # --------------------------------

    velocity_error = (
        velocities -
        desired_velocities
    )


    # --------------------------------
    # Graph coupling
    # --------------------------------

    graph_error = distributed_error(
        position_error,
        L
    )


    # --------------------------------
    # Nominal acceleration
    # --------------------------------

    controls = (

        leader_acceleration[None, :]

        - config.KP *
        position_error

        - config.KV *
        velocity_error

        - config.K_GRAPH *
        graph_error
    )


    # --------------------------------
    # Acceleration saturation
    # --------------------------------

    norms = np.linalg.norm(
        controls,
        axis=1,
        keepdims=True
    )


    scale = np.minimum(
        1.0,
        config.MAX_ACCELERATION /
        np.maximum(norms, 1e-12)
    )


    controls = (
        controls *
        scale
    )


    return (
        controls,
        position_error,
        velocity_error
    )