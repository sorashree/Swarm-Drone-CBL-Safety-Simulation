"""
Main swarm simulation loop.
"""

import numpy as np

import config

from drones import Drone

from formation import (
    desired_formation
)

from graph import (
    graph_matrices
)

from controller import (
    nominal_control
)

from cbf import (
    cbf_filter
)

from dynamics import (
    step
)

from metrics import (
    rms_formation_error,
    minimum_separation
)


def create_initial_drones():

    """
    Create drones with small initial errors.
    """

    rng = np.random.default_rng(
        config.SEED
    )


    (
        desired_positions,
        desired_velocities,
        _,
        _,
        _
    ) = desired_formation(0.0)


    drones = []


    for i in range(
        config.NUM_DRONES
    ):

        position = (

            desired_positions[i]

            + rng.normal(
                0.0,
                config.INITIAL_POSITION_NOISE,
                size=2
            )
        )


        velocity = (

            desired_velocities[i]

            + rng.normal(
                0.0,
                config.INITIAL_VELOCITY_NOISE,
                size=2
            )
        )


        drones.append(

            Drone(

                drone_id=i,

                position=position,

                velocity=velocity
            )
        )


    return drones


def run_simulation():

    """
    Run complete swarm simulation.
    """

    steps = int(
        config.SIM_TIME /
        config.DT
    )


    # --------------------------------
    # Communication graph
    # --------------------------------

    A, D, L = graph_matrices(
        config.NUM_DRONES
    )


    # --------------------------------
    # Initial drones
    # --------------------------------

    drones = (
        create_initial_drones()
    )


    # --------------------------------
    # Storage
    # --------------------------------

    times = []

    position_history = []

    velocity_history = []

    desired_history = []

    leader_history = []

    nominal_history = []

    control_history = []

    rms_history = []

    separation_history = []

    cbf_active_history = []

    cbf_success_history = []


    # =================================
    # MAIN LOOP
    # =================================

    for k in range(
        steps + 1
    ):

        t = (
            k *
            config.DT
        )


        # Current states
        positions = np.array([
            d.position
            for d in drones
        ])


        velocities = np.array([
            d.velocity
            for d in drones
        ])


        # --------------------------------
        # Desired formation
        # --------------------------------

        (
            desired_positions,
            desired_velocities,
            leader_acceleration,
            leader_position,
            _
        ) = desired_formation(t)


        # --------------------------------
        # Nominal distributed controller
        # --------------------------------

        (
            nominal,
            _,
            _
        ) = nominal_control(

            positions,

            velocities,

            desired_positions,

            desired_velocities,

            leader_acceleration,

            L
        )


        # --------------------------------
        # CBF safety filter
        # --------------------------------

        (
            safe_controls,
            cbf_info
        ) = cbf_filter(

            positions,

            velocities,

            nominal
        )


        # --------------------------------
        # Save state
        # --------------------------------

        times.append(t)

        position_history.append(
            positions.copy()
        )

        velocity_history.append(
            velocities.copy()
        )

        desired_history.append(
            desired_positions.copy()
        )

        leader_history.append(
            leader_position.copy()
        )

        nominal_history.append(
            nominal.copy()
        )

        control_history.append(
            safe_controls.copy()
        )


        # --------------------------------
        # Metrics
        # --------------------------------

        rms_history.append(

            rms_formation_error(

                positions,

                desired_positions
            )
        )


        separation_history.append(

            minimum_separation(
                positions
            )
        )


        cbf_active_history.append(

            cbf_info[
                "active_or_near_cbf_constraints"
            ]
        )


        cbf_success_history.append(

            cbf_info[
                "qp_success"
            ]
        )


        # --------------------------------
        # Don't integrate after final frame
        # --------------------------------

        if k == steps:

            break


        # --------------------------------
        # Update every drone
        # --------------------------------

        for i, drone in enumerate(
            drones
        ):

            (
                new_position,
                new_velocity
            ) = step(

                drone.position,

                drone.velocity,

                safe_controls[i],

                config.DT
            )


            drone.position = (
                new_position
            )

            drone.velocity = (
                new_velocity
            )


    return {

        "time":
            np.asarray(times),

        "positions":
            np.asarray(
                position_history
            ),

        "velocities":
            np.asarray(
                velocity_history
            ),

        "desired_positions":
            np.asarray(
                desired_history
            ),

        "leader_positions":
            np.asarray(
                leader_history
            ),

        "nominal_controls":
            np.asarray(
                nominal_history
            ),

        "controls":
            np.asarray(
                control_history
            ),

        "rms_error":
            np.asarray(
                rms_history
            ),

        "minimum_separation":
            np.asarray(
                separation_history
            ),

        "cbf_active":
            np.asarray(
                cbf_active_history
            ),

        "cbf_success":
            np.asarray(
                cbf_success_history
            ),

        "adjacency": A,

        "degree": D,

        "laplacian": L
    }