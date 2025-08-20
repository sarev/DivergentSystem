#!/usr/bin/env python3

from LorenzSystem import LorenzSystem
from RosslerSystem import RosslerSystem
from ChenSystem import ChenSystem
from HalvorsenSystem import HalvorsenSystem
from ShimizuMoriokaSystem import ShimizuMoriokaSystem
from DualPendulumSystem1 import DualPendulumSystem1
from typing import Tuple, Any


def plot_trajectory(
        system: Any,
        config: Tuple,
        origin: Tuple,
        steps: int = 20000,
        block: bool = False
    ) -> None:

    system.config = config
    trajectory = system.get_trajectory(origin, steps)
    return system.plot_trajectory(trajectory, origin, block=block)


def main():
    # Create some system instances
    chen = ChenSystem()
    dualpendulum = DualPendulumSystem1()
    halvorsen = HalvorsenSystem()
    lorenz = LorenzSystem()
    rossler = RosslerSystem()
    shimizumorioka = ShimizuMoriokaSystem()

    # Create some trajectories and plot them in windows
    plots = []
    plots.append(plot_trajectory(chen, (35.0, 3.0, 28.0), (1.0, 0.0, 0.0)))
    plots.append(plot_trajectory(dualpendulum, (0, 0, 1.0, 9.80665), (1.7627825445142729, 2.007128639793479, 0.0)))
    plots.append(plot_trajectory(halvorsen, (1.4), (0.1, 0.0, 0.0)))
    plots.append(plot_trajectory(lorenz, (10.0, 28.0, 8.0 / 3.0), (1.0, 1.0, 1.0)))
    plots.append(plot_trajectory(rossler, (0.2, 0.2, 5.7), (1.0, 0.0, 0.0)))
    plots.append(plot_trajectory(shimizumorioka, (0.81, 0.375), (0.1, 0.2, 0.1), block=True))

    # Tidy up - by way of an example
    for p in plots:
        p.close()


if __name__ == "__main__":
    main()
