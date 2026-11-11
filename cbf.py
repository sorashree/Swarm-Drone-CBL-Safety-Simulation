"""
Control Barrier Function safety filter.

For every drone pair:

    h_ij = ||p_i-p_j||^2 - d_s^2

Safety requires:

    h_ij >= 0

For second-order dynamics we use:

    h_ddot + alpha1*h_dot + alpha0*h >= 0

This produces a linear constraint on accelerations.

The safety filter solves:

    minimize 0.5 ||U - U_nom||^2

subject to all CBF constraints.
"""

import numpy as np

from scipy.optimize import (
    minimize,
    LinearConstraint
)

import config


def pairwise_constraint_row(
    positions,
    velocities,
    i,
    j
):

    """
    Build one CBF constraint.

    Returns:

        row
        rhs
        h
        distance

    such that:

        row @ U >= rhs
    """

    n = len(positions)


    row = np.zeros(
        2 * n,
        dtype=float
    )


    # Relative position
    r = (
        positions[i] -
        positions[j]
    )


    # Relative velocity
    v = (
        velocities[i] -
        velocities[j]
    )


    distance_squared = float(
        np.dot(r, r)
    )


    distance = np.sqrt(
        max(distance_squared, 0.0)
    )


    # --------------------------------
    # Barrier function
    # --------------------------------

    h = (
        distance_squared -
        config.SAFETY_DISTANCE**2
    )


    # --------------------------------
    # First derivative
    # --------------------------------

    h_dot = (
        2.0 *
        float(np.dot(r, v))
    )


    # --------------------------------
    # Second derivative
    #
    # h_ddot =
    #
    # 2 v^T v
    #
    # + 2 r^T(u_i-u_j)
    # --------------------------------

    constant_term = (
        2.0 *
        float(np.dot(v, v))
    )


    rhs = (

        -constant_term

        - (
            config.CBF_ALPHA1 + config.CBF_ALPHA2
        )* h_dot

        - (
            config.CBF_ALPHA1 * config.CBF_ALPHA2
        ) * h
    )


    # Acceleration of drone i
    row[
        2*i:
        2*i + 2
    ] = 2.0 * r


    # Acceleration of drone j
    row[
        2*j:
        2*j + 2
    ] = -2.0 * r


    return (
        row,
        rhs,
        h,
        distance
    )


def cbf_filter(
    positions,
    velocities,
    nominal_controls
):

    """
    Find the closest safe acceleration
    to the nominal controller command.
    """

    n = len(positions)


    # Flatten:

    # [u1x,u1y,u2x,u2y,...]

    u_nom = (
        nominal_controls
        .reshape(-1)
        .astype(float)
    )


    rows = []

    lower_bounds = []

    distances = []

    h_values = []


    # --------------------------------
    # Create constraints for every pair
    # --------------------------------

    for i in range(n):

        for j in range(i + 1, n):

            (
                row,
                rhs,
                h,
                distance
            ) = pairwise_constraint_row(
                positions,
                velocities,
                i,
                j
            )


            rhs += (
                config.CBF_SAFETY_MARGIN
            )


            rows.append(row)

            lower_bounds.append(rhs)

            distances.append(distance)

            h_values.append(h)


    A = np.vstack(rows)


    # --------------------------------
    # Objective
    #
    # Keep control as close as possible
    # to nominal controller.
    # --------------------------------

    def objective(U):

        difference = (
            U - u_nom
        )

        return 0.5 * np.dot(
            difference,
            difference
        )


    def gradient(U):

        return U - u_nom


    # --------------------------------
    # Acceleration bounds
    # --------------------------------

    max_acceleration = (
        config.MAX_ACCELERATION /
        np.sqrt(2.0)
    )


    bounds = [
        (
            -max_acceleration,
            max_acceleration
        )
        for _ in range(2*n)
    ]


    # --------------------------------
    # CBF:
    #
    # A U >= lower_bounds
    # --------------------------------

    constraints = LinearConstraint(
        A,
        np.asarray(lower_bounds),
        np.full(
            len(lower_bounds),
            np.inf
        )
    )


    # --------------------------------
    # Solve QP
    # --------------------------------

    result = minimize(

        objective,

        u_nom,

        jac=gradient,

        method="SLSQP",

        bounds=bounds,

        constraints=[
            constraints
        ],

        options={
            "ftol": 1e-8,
            "maxiter": 200,
            "disp": False
        }
    )


    if result.success:

        safe_controls = (
            result.x
            .reshape(n, 2)
        )

    else:

        # Emergency fallback
        safe_controls = (
            nominal_controls.copy()
        )


    # --------------------------------
    # Diagnostics
    # --------------------------------

    active = int(
        sum(
            h <= 0.15
            for h in h_values
        )
    )


    diagnostics = {

        "qp_success":
            bool(result.success),

        "qp_message":
            str(result.message),

        "minimum_current_separation":
            float(min(distances)),

        "active_or_near_cbf_constraints":
            active,

        "minimum_h":
            float(min(h_values))
    }


    return (
        safe_controls,
        diagnostics
    )
