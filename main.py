"""
Entry point for the swarm-drone project.
"""

from pathlib import Path

import config

from simulation import (
    run_simulation
)

from visualization import (
    create_animation,
    create_result_plot
)


def main():

    Path(
        config.OUTPUT_DIR
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 70)

    print(
        "8-DRONE DISTRIBUTED FORMATION "
        "CONTROL + CBF SAFETY SIMULATION"
    )

    print("=" * 70)


    # --------------------------------
    # Run simulation
    # --------------------------------

    data = run_simulation()


    # --------------------------------
    # Generate PNG
    # --------------------------------

    result_file = (
        create_result_plot(data)
    )


    # --------------------------------
    # Generate MP4
    # --------------------------------

    try:

        animation_file = (
            create_animation(data)
        )

    except Exception as exc:

        animation_file = None

        print(
            "\nAnimation generation failed."
        )

        print(
            "Make sure FFmpeg is installed "
            "and available on PATH."
        )

        print(
            f"Reason: {exc}"
        )


    final = -1


    # =================================
    # RESULTS
    # =================================

    print("\nSIMULATION RESULTS")

    print("-" * 70)


    print(
        f"Number of drones: "
        f"{config.NUM_DRONES}"
    )


    print(
        f"Formation: "
        f"regular {config.NUM_DRONES}-gon"
    )


    print(
        "Communication topology: ring"
    )


    print(
        f"Safety distance: "
        f"{config.SAFETY_DISTANCE:.2f} m"
    )


    print(
        f"Minimum observed separation: "
        f"{data['minimum_separation'].min():.3f} m"
    )


    print(
        f"Final RMS formation error: "
        f"{data['rms_error'][final]:.3f} m"
    )


    print(
        f"Final CBF QP success: "
        f"{bool(data['cbf_success'][final])}"
    )


    print("\nOUTPUTS")

    print("-" * 70)


    print(
        f"Static result: "
        f"{result_file}"
    )


    if animation_file:

        print(
            f"Animation: "
            f"{animation_file}"
        )


    print("\nDone.")


if __name__ == "__main__":

    main()
