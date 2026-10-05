"""
Drone state representation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Drone:

    drone_id: int

    position: np.ndarray

    velocity: np.ndarray


    def copy(self):

        return Drone(
            self.drone_id,
            self.position.copy(),
            self.velocity.copy()
        )
