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
    Initialise UAVs at scattered positions around
    the virtual leader with zero initial velocity.
    """

    rng = np.random.default_rng(config.SEED)

    drones = []

    base_angles = np.linspace(
        0.0,
        2.0 * np.pi,
        config.NUM_DRONES,
        endpoint=False
    )

    angle_variations = rng.uniform(
        -0.18,
        0.18,
        config.NUM_DRONES
    )

    starting_radii = rng.uniform(
        4.5,
        5.2,
        config.NUM_DRONES
    )

    starting_positions = []

    for i in range(config.NUM_DRONES):

        angle = (
            base_angles[i]
            + angle_variations[i]
        )

        radius = starting_radii[i]

        position = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle)
        ])

        starting_positions.append(position)

    # Validate initial separation.
    for i in range(config.NUM_DRONES):

        for j in range(i + 1, config.NUM_DRONES):

            distance = np.linalg.norm(
                starting_positions[i]
                - starting_positions[j]
            )

            if distance < config.SAFETY_DISTANCE:

                raise ValueError(
                    f"Initial separation violation: "
                    f"UAV {i + 1} and UAV {j + 1}: "
                    f"{distance:.3f} m"
                )

    for i in range(config.NUM_DRONES):

        drones.append(
            Drone(
                drone_id=i,
                position=starting_positions[i],
                velocity=np.zeros(2)
            )
        )

    return drones



def run_simulation(use_cbf=True):

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

    cbf_intervention_history = []


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

        if use_cbf:

            (
                safe_controls,
                cbf_info
            ) = cbf_filter(

                positions,

                velocities,

                nominal
            )

        else:

            # Baseline: use nominal controller directly.
            safe_controls = nominal.copy()

            cbf_info = None


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


        if use_cbf:

            cbf_active_history.append(
                cbf_info["active_or_near_cbf_constraints"]
            )

            cbf_success_history.append(
                cbf_info["qp_success"]
            )

        else:

            # NaN means the CBF/QP was not run.
            cbf_active_history.append(np.nan)
            cbf_success_history.append(np.nan)


        if use_cbf:

            # Fraction of drones whose commands were changed by CBF.
            changed = (
                np.linalg.norm(
                    safe_controls - nominal,
                    axis=1
                ) > 1e-6
            )

            cbf_intervention_history.append(
                float(np.mean(changed))
            )

        else:

            # CBF was not executed: intervention is not applicable.
            cbf_intervention_history.append(np.nan)


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

        "laplacian": L,

        "cbf_intervention_fraction": np.asarray(
            cbf_intervention_history
        ),

        "cbf_enabled": bool(use_cbf)
    }
