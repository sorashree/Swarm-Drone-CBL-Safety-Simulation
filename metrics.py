"""
Swarm performance metrics.
"""

import numpy as np


def rms_formation_error(
    positions,
    desired_positions
):

    """
    RMS Euclidean formation error.
    """

    errors = (
        positions -
        desired_positions
    )


    squared = np.sum(
        errors**2,
        axis=1
    )


    return float(
        np.sqrt(
            np.mean(squared)
        )
    )


def minimum_separation(
    positions
):

    """
    Find the closest pair of drones.
    """

    n = len(positions)

    minimum = np.inf


    for i in range(n):

        for j in range(i + 1, n):

            distance = np.linalg.norm(
                positions[i] -
                positions[j]
            )


            minimum = min(
                minimum,
                distance
            )


    return float(minimum)


def formation_rms_history(
    position_history,
    desired_history
):

    return np.array([

        rms_formation_error(
            p,
            d
        )

        for p, d in zip(
            position_history,
            desired_history
        )

    ])


def separation_history(
    position_history
):

    return np.array([

        minimum_separation(p)

        for p in position_history

    ])
