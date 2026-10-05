"""
Visualization for the 8-drone distributed formation-control simulation.

Features
--------
- Animated 8-drone swarm
- Mathematical quadcopter/drone icons
- Desired formation positions
- Virtual leader
- Dashed virtual-leader figure-eight trajectory
- Ring communication topology
- No individual drone movement trails
- RMS formation error
- Minimum separation
- CBF safety information
- MP4 animation
- Static result image
"""

import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


# ============================================================
# FFmpeg configuration
# ============================================================

mpl.rcParams["animation.ffmpeg_path"] = (
    r"C:\Users\Shreeya Mukherjee\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.2-full_build-shared"
    r"\bin\ffmpeg.exe"
)


# ============================================================
# Utility
# ============================================================

def ensure_output_directory():
    """Create output directory if it does not exist."""

    os.makedirs("outputs", exist_ok=True)


# ============================================================
# Mathematical Quadcopter
# ============================================================

class DroneArtist:
    """
    Draw one mathematical quadcopter.

    Each drone contains:

        rotor ----- arm ----- rotor
                    |
                  body
                    |
        rotor ----- arm ----- rotor

    The entire drone moves and rotates according to
    the simulated position and velocity.
    """

    def __init__(
        self,
        ax,
        position,
        angle=0.0,
        scale=0.30
    ):

        self.ax = ax
        self.scale = scale

        self.position = np.asarray(
            position,
            dtype=float
        )

        self.angle = float(angle)

        # ----------------------------------------------------
        # Central body
        # ----------------------------------------------------

        self.body = Circle(
            self.position,
            radius=0.11,
            zorder=9
        )

        ax.add_patch(self.body)

        # ----------------------------------------------------
        # Four rotor circles
        # ----------------------------------------------------

        self.rotors = []

        for _ in range(4):

            rotor = Circle(
                self.position,
                radius=0.075,
                fill=False,
                linewidth=1.5,
                zorder=8
            )

            ax.add_patch(rotor)

            self.rotors.append(rotor)

        # ----------------------------------------------------
        # Four arms
        # ----------------------------------------------------

        self.arms = []

        for _ in range(4):

            arm = Line2D(
                [],
                [],
                linewidth=2.2,
                zorder=7
            )

            ax.add_line(arm)

            self.arms.append(arm)

        # ----------------------------------------------------
        # Direction indicator
        # ----------------------------------------------------

        self.direction = Line2D(
            [],
            [],
            linewidth=1.8,
            zorder=10
        )

        ax.add_line(self.direction)

        # Initial drawing
        self.update(
            self.position,
            self.angle
        )

    # ========================================================
    # Rotate local point
    # ========================================================

    def rotate_point(self, point):

        c = np.cos(self.angle)
        s = np.sin(self.angle)

        rotation = np.array([
            [c, -s],
            [s,  c]
        ])

        point = np.asarray(
            point,
            dtype=float
        )

        return (
            self.position
            + rotation @ point
        )

    # ========================================================
    # Update drone
    # ========================================================

    def update(
        self,
        position,
        angle=0.0
    ):

        self.position = np.asarray(
            position,
            dtype=float
        )

        self.angle = float(angle)

        # ----------------------------------------------------
        # Drone geometry
        # ----------------------------------------------------

        arm_length = self.scale

        # Four rotor locations in local coordinates
        local_rotors = np.array([
            [ arm_length,  arm_length],
            [-arm_length,  arm_length],
            [-arm_length, -arm_length],
            [ arm_length, -arm_length]
        ])

        world_rotors = []

        # ----------------------------------------------------
        # Update rotors
        # ----------------------------------------------------

        for i, local_point in enumerate(
            local_rotors
        ):

            world_point = self.rotate_point(
                local_point
            )

            world_rotors.append(
                world_point
            )

            self.rotors[i].center = (
                world_point
            )

        # ----------------------------------------------------
        # Update four arms
        # ----------------------------------------------------

        for i, rotor_position in enumerate(
            world_rotors
        ):

            self.arms[i].set_data(
                [
                    self.position[0],
                    rotor_position[0]
                ],
                [
                    self.position[1],
                    rotor_position[1]
                ]
            )

        # ----------------------------------------------------
        # Update central body
        # ----------------------------------------------------

        self.body.center = (
            self.position
        )

        # ----------------------------------------------------
        # Update direction indicator
        # ----------------------------------------------------

        direction_length = (
            arm_length * 0.55
        )

        direction_point = self.rotate_point(
            [
                direction_length,
                0.0
            ]
        )

        self.direction.set_data(
            [
                self.position[0],
                direction_point[0]
            ],
            [
                self.position[1],
                direction_point[1]
            ]
        )

    # ========================================================
    # Return all artists
    # ========================================================

    def artists(self):

        return (
            [self.body]
            + self.rotors
            + self.arms
            + [self.direction]
        )


# ============================================================
# Drone heading
# ============================================================

def calculate_drone_heading(
    velocity,
    fallback_angle=0.0
):
    """
    Calculate drone heading from velocity.

    If velocity is almost zero, use fallback_angle.
    """

    velocity = np.asarray(
        velocity,
        dtype=float
    )

    speed = np.linalg.norm(
        velocity
    )

    if speed < 1e-8:

        return fallback_angle

    return np.arctan2(
        velocity[1],
        velocity[0]
    )


# ============================================================
# Create animation
# ============================================================

def create_animation(data):
    """
    Create animated swarm simulation.

    Individual drone movement trails are intentionally
    NOT drawn to keep the visualization clean.
    """

    ensure_output_directory()

    # ========================================================
    # Extract simulation data
    # ========================================================

    time = np.asarray(
        data["time"]
    )

    positions = np.asarray(
        data["positions"]
    )

    velocities = np.asarray(
        data["velocities"]
    )

    desired_positions = np.asarray(
        data["desired_positions"]
    )

    leader_positions = np.asarray(
        data["leader_positions"]
    )

    rms_error = np.asarray(
        data["rms_error"]
    )

    minimum_separation = np.asarray(
        data["minimum_separation"]
    )

    cbf_active = np.asarray(
        data["cbf_active"]
    )

    safety_distance = 1.20

    num_frames = len(time)

    num_drones = positions.shape[1]

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.set_title(
        "Distributed Swarm-Drone Formation Control "
        "with CBF Collision Avoidance"
    )

    ax.set_xlabel(
        "x position [m]"
    )

    ax.set_ylabel(
        "y position [m]"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    # ========================================================
    # Calculate plot limits
    # ========================================================

    all_positions = np.concatenate(
        [
            positions.reshape(-1, 2),
            desired_positions.reshape(-1, 2),
            leader_positions.reshape(-1, 2)
        ],
        axis=0
    )

    min_x = np.min(
        all_positions[:, 0]
    )

    max_x = np.max(
        all_positions[:, 0]
    )

    min_y = np.min(
        all_positions[:, 1]
    )

    max_y = np.max(
        all_positions[:, 1]
    )

    margin = 2.0

    ax.set_xlim(
        min_x - margin,
        max_x + margin
    )

    ax.set_ylim(
        min_y - margin,
        max_y + margin
    )

    # ========================================================
    # Virtual leader complete figure-eight path
    # ========================================================

    ax.plot(
        leader_positions[:, 0],
        leader_positions[:, 1],
        linestyle="--",
        linewidth=1.8,
        alpha=0.75,
        label="Virtual-leader path",
        zorder=1
    )

    # ========================================================
    # Desired formation positions
    # ========================================================

    desired_scatter = ax.scatter(
        desired_positions[0, :, 0],
        desired_positions[0, :, 1],
        s=90,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        zorder=4,
        label="Desired positions"
    )

    # ========================================================
    # Ring communication topology
    # ========================================================

    communication_lines = []

    for i in range(
        num_drones
    ):

        j = (
            i + 1
        ) % num_drones

        line, = ax.plot(
            [],
            [],
            linestyle="-",
            linewidth=0.8,
            alpha=0.30,
            zorder=3
        )

        communication_lines.append(
            (
                line,
                i,
                j
            )
        )

    # ========================================================
    # Create mathematical drones
    # ========================================================

    drones = []

    for i in range(
        num_drones
    ):

        drone = DroneArtist(
            ax,
            positions[0, i],
            angle=0.0,
            scale=0.30
        )

        drones.append(
            drone
        )

    # ========================================================
    # Virtual leader marker
    # ========================================================

    leader_marker, = ax.plot(
        [],
        [],
        marker="*",
        markersize=14,
        linestyle="None",
        zorder=12,
        label="Virtual leader"
    )

    # ========================================================
    # Safety circles
    # ========================================================

    safety_circles = []

    for i in range(
        num_drones
    ):

        circle = Circle(
            positions[0, i],
            radius=safety_distance / 2.0,
            fill=False,
            linestyle=":",
            linewidth=0.7,
            alpha=0.15,
            zorder=2
        )

        ax.add_patch(
            circle
        )

        safety_circles.append(
            circle
        )

    # ========================================================
    # Information box
    # ========================================================

    info_box = ax.text(
        0.02,
        0.02,
        "",
        transform=ax.transAxes,
        verticalalignment="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            alpha=0.8
        ),
        zorder=20
    )

    # ========================================================
    # Controller equation box
    # ========================================================

    equation_box = ax.text(
        0.02,
        0.96,
        r"$\dot p_i=v_i,\quad \dot v_i=u_i$"
        "\n"
        r"$u_i^{nom}=a_c-K_p e_i-K_v\dot e_i-K_gLe$",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            alpha=0.8
        ),
        zorder=20
    )

    # ========================================================
    # CBF equation box
    # ========================================================

    cbf_box = ax.text(
        0.98,
        0.02,
        r"$h_{ij}=\|p_i-p_j\|^2-d^2\geq0$"
        "\n"
        r"$\dot h+\alpha h\geq0$",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            alpha=0.8
        ),
        zorder=20
    )

    # ========================================================
    # Legend
    # ========================================================

    drone_legend = Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=8,
        label="Actual drones"
    )

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    handles.insert(
        1,
        drone_legend
    )

    labels.insert(
        1,
        "Actual drones"
    )

    ax.legend(
        handles,
        labels,
        loc="upper right"
    )

    # ========================================================
    # Animation initialization
    # ========================================================

    def init():

        # Clear communication links
        for line, _, _ in (
            communication_lines
        ):

            line.set_data(
                [],
                []
            )

        # Clear leader marker
        leader_marker.set_data(
            [],
            []
        )

        return []

    # ========================================================
    # Animation update
    # ========================================================

    def update(frame):

        # ----------------------------------------------------
        # Current simulation state
        # ----------------------------------------------------

        current_positions = (
            positions[frame]
        )

        current_velocities = (
            velocities[frame]
        )

        current_desired = (
            desired_positions[frame]
        )

        current_leader = (
            leader_positions[frame]
        )

        # ----------------------------------------------------
        # Desired positions
        # ----------------------------------------------------

        desired_scatter.set_offsets(
            current_desired
        )

        # ----------------------------------------------------
        # Communication ring
        # ----------------------------------------------------

        for line, i, j in (
            communication_lines
        ):

            line.set_data(
                [
                    current_positions[i, 0],
                    current_positions[j, 0]
                ],
                [
                    current_positions[i, 1],
                    current_positions[j, 1]
                ]
            )

        # ----------------------------------------------------
        # Update drones
        # ----------------------------------------------------

        for i, drone in enumerate(
            drones
        ):

            angle = (
                calculate_drone_heading(
                    current_velocities[i]
                )
            )

            drone.update(
                current_positions[i],
                angle
            )

        # ----------------------------------------------------
        # Update safety circles
        # ----------------------------------------------------

        for i, circle in enumerate(
            safety_circles
        ):

            circle.center = (
                current_positions[i]
            )

        # ----------------------------------------------------
        # Update virtual leader
        # ----------------------------------------------------

        leader_marker.set_data(
            [
                current_leader[0]
            ],
            [
                current_leader[1]
            ]
        )

        # ----------------------------------------------------
        # Current metrics
        # ----------------------------------------------------

        current_rms = (
            rms_error[frame]
        )

        current_min_sep = (
            minimum_separation[frame]
        )

        current_cbf = (
            cbf_active[frame]
        )

        # ----------------------------------------------------
        # Information box
        # ----------------------------------------------------

        info_box.set_text(
            f"Simulated time: "
            f"{time[frame]:.2f} s\n"
            f"RMS formation error: "
            f"{current_rms:.3f} m\n"
            f"Minimum separation: "
            f"{current_min_sep:.3f} m\n"
            f"Safety threshold: "
            f"{safety_distance:.2f} m\n"
            f"CBF near-active constraints: "
            f"{int(current_cbf)}"
        )

        return []

    # ========================================================
    # Create animation
    # ========================================================

    animation = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=num_frames,
        interval=20,
        blit=False,
        repeat=False
    )

    # ========================================================
    # Output path
    # ========================================================

    output_path = os.path.join(
        "outputs",
        "swarm_animation.mp4"
    )

    # ========================================================
    # FFmpeg writer
    # ========================================================

    writer = FFMpegWriter(
        fps=30,
        metadata={
            "title":
                "8-Drone Distributed Formation Control"
        },
        bitrate=2500
    )

    # ========================================================
    # Save animation
    # ========================================================

    animation.save(
        output_path,
        writer=writer,
        dpi=120
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# Static result plot
# ============================================================

def create_result_plot(data):
    """
    Create a clean static summary.

    IMPORTANT:
    Individual drone movement trajectories are NOT drawn.
    Only the virtual-leader figure-eight path is shown.
    """

    ensure_output_directory()

    # ========================================================
    # Extract data
    # ========================================================

    positions = np.asarray(
        data["positions"]
    )

    desired_positions = np.asarray(
        data["desired_positions"]
    )

    leader_positions = np.asarray(
        data["leader_positions"]
    )

    rms_error = np.asarray(
        data["rms_error"]
    )

    minimum_separation = np.asarray(
        data["minimum_separation"]
    )

    velocities = np.asarray(
        data["velocities"]
    )

    safety_distance = 1.20

    num_drones = positions.shape[1]

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.set_title(
        "8-Drone Distributed Formation Control "
        "with CBF Collision Avoidance"
    )

    ax.set_xlabel(
        "x position [m]"
    )

    ax.set_ylabel(
        "y position [m]"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    # ========================================================
    # Virtual leader figure-eight path
    # ========================================================

    ax.plot(
        leader_positions[:, 0],
        leader_positions[:, 1],
        linestyle="--",
        linewidth=2.0,
        alpha=0.75,
        label="Virtual-leader path",
        zorder=1
    )

    # ========================================================
    # Ring communication topology
    # ========================================================

    final_positions = (
        positions[-1]
    )

    for i in range(
        num_drones
    ):

        j = (
            i + 1
        ) % num_drones

        ax.plot(
            [
                final_positions[i, 0],
                final_positions[j, 0]
            ],
            [
                final_positions[i, 1],
                final_positions[j, 1]
            ],
            linewidth=0.9,
            alpha=0.30,
            zorder=2
        )

    # ========================================================
    # Final desired positions
    # ========================================================

    ax.scatter(
        desired_positions[-1, :, 0],
        desired_positions[-1, :, 1],
        s=100,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Desired positions",
        zorder=5
    )

    # ========================================================
    # Final drone positions
    # ========================================================

    for i in range(
        num_drones
    ):

        final_velocity = (
            velocities[-1, i]
        )

        angle = (
            calculate_drone_heading(
                final_velocity
            )
        )

        DroneArtist(
            ax,
            final_positions[i],
            angle=angle,
            scale=0.30
        )

    # ========================================================
    # Final virtual leader
    # ========================================================

    ax.plot(
        leader_positions[-1, 0],
        leader_positions[-1, 1],
        marker="*",
        markersize=15,
        linestyle="None",
        label="Virtual leader",
        zorder=12
    )

    # ========================================================
    # Metrics
    # ========================================================

    minimum_observed = np.min(
        minimum_separation
    )

    final_rms = (
        rms_error[-1]
    )

    ax.text(
        0.02,
        0.02,
        f"RMS formation error: "
        f"{final_rms:.3f} m\n"
        f"Minimum separation: "
        f"{minimum_observed:.3f} m\n"
        f"Safety threshold: "
        f"{safety_distance:.2f} m",
        transform=ax.transAxes,
        verticalalignment="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            alpha=0.8
        )
    )

    # ========================================================
    # Legend
    # ========================================================

    ax.legend(
        loc="upper right"
    )

    # ========================================================
    # Save
    # ========================================================

    output_path = os.path.join(
        "outputs",
        "swarm_result.png"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path
