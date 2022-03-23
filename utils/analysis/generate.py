import numpy as np
from rlgym.utils import common_values


def grid_positions(point_distance=1600, quarter_split=None):
    """
    A method that generates equidistant positions in a grid for the Rocket League arena

    :param point_distance: Distance between neighbouring points
    :param quarter_split: A factor by which to split x, y and z axis positive values in the arena.
        If specified, axes are split into an equal number of partitions, breaking, however, the equidistance property.
    :return: A numpy array containing 3d arena positions in a grid
    """
    if quarter_split:
        splits = (2 * quarter_split + 1,) * 2 + (quarter_split + 1,)
        end_points = ((common_values.SIDE_WALL_X, -common_values.SIDE_WALL_X),
                      (common_values.BACK_WALL_Y, -common_values.BACK_WALL_Y),
                      (common_values.CEILING_Z, 0))
    else:
        splits = (2 * common_values.SIDE_WALL_X // point_distance + 1,
                  2 * common_values.BACK_WALL_Y // point_distance + 1,
                  common_values.CEILING_Z // point_distance + 1)
        x_mod = common_values.SIDE_WALL_X % point_distance
        y_mod = common_values.BACK_WALL_Y % point_distance
        z_mod = (common_values.CEILING_Z % point_distance) // 2
        end_points = ((common_values.SIDE_WALL_X - x_mod, -common_values.SIDE_WALL_X + x_mod),
                      (common_values.BACK_WALL_Y - y_mod, -common_values.BACK_WALL_Y + y_mod),
                      (common_values.CEILING_Z - z_mod, 0 + z_mod))

    xs = np.linspace(end_points[0][0], end_points[0][1], splits[0])
    ys = np.linspace(end_points[1][0], end_points[1][1], splits[1])
    zs = np.linspace(end_points[2][0], end_points[2][1], splits[2])

    positions = np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3)
    abs_positions = np.abs(positions)
    not_beyond_corner = (abs_positions[:, 0] + abs_positions[:, 1] <
                         common_values.SIDE_WALL_X + common_values.BACK_WALL_Y - 1152)
    return positions[not_beyond_corner]


def sphere_points(n_points=50, radius=1):
    """
    Generates a number of equidistant neighbouring points on a sphere

    Taken from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.stack([x, y, z], axis=-1) * radius
