"""
Communication graph for the distributed swarm controller.
"""

import numpy as np
import config


def ring_adjacency(
    n=config.NUM_DRONES
):

    """
    Create an undirected ring.

    Example:

    D1 -- D2
    |     |
    D8    D3
    |     |
    D7 -- D4
       ...
    """

    A = np.zeros(
        (n, n),
        dtype=float
    )


    for i in range(n):

        # Previous drone
        A[
            i,
            (i - 1) % n
        ] = 1.0


        # Next drone
        A[
            i,
            (i + 1) % n
        ] = 1.0


    return A


def graph_matrices(
    n=config.NUM_DRONES
):

    """
    Calculate:

        A = adjacency matrix
        D = degree matrix
        L = graph Laplacian

    L = D - A
    """

    A = ring_adjacency(n)


    D = np.diag(
        A.sum(axis=1)
    )


    L = D - A


    return A, D, L


def distributed_error(
    errors,
    L
):

    """
    Calculate graph-coupled error.

    errors shape:

        (number_of_drones, 2)
    """

    return L @ errors