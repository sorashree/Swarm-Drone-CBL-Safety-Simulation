"""
Discrete-time second-order point-mass dynamics.
"""

import numpy as np
import config


def saturate_vector(
    vector,
    maximum
):

    """
    Limit vector magnitude.
    """

    norm = np.linalg.norm(vector)


    if norm <= maximum:

        return vector.copy()


    if norm == 0:

        return vector.copy()


    return vector * (
        maximum / norm
    )


def step(
    position,
    velocity,
    acceleration,
    dt=config.DT
):

    """
    Second-order point-mass model.

    v(k+1) = v(k) + a(k)dt

    p(k+1) = p(k) + v(k+1)dt
    """

    acceleration = saturate_vector(
        np.asarray(
            acceleration,
            dtype=float
        ),
        config.MAX_ACCELERATION
    )


    new_velocity = (
        velocity +
        acceleration * dt
    )


    new_velocity = saturate_vector(
        new_velocity,
        config.MAX_SPEED
    )


    new_position = (
        position +
        new_velocity * dt
    )


    return (
        new_position,
        new_velocity
    )