from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri
from rlgym.utils import common_values
from scipy.spatial.kdtree import KDTree


def make_arena_(refiner_sub_div=6):
    corner_offset = 1152
    right_corner0 = np.array([[- common_values.SIDE_WALL_X + corner_offset, - common_values.BACK_WALL_Y],
                              [- common_values.SIDE_WALL_X, - common_values.BACK_WALL_Y + corner_offset]])
    left_corner0 = np.array([[common_values.SIDE_WALL_X - corner_offset, - common_values.BACK_WALL_Y],
                             [common_values.SIDE_WALL_X, - common_values.BACK_WALL_Y + corner_offset]])
    right_corner1 = np.array([[- common_values.SIDE_WALL_X + corner_offset, common_values.BACK_WALL_Y],
                              [- common_values.SIDE_WALL_X, common_values.BACK_WALL_Y - corner_offset]])
    left_corner1 = np.array([[common_values.SIDE_WALL_X - corner_offset, common_values.BACK_WALL_Y],
                             [common_values.SIDE_WALL_X, common_values.BACK_WALL_Y - corner_offset]])

    x = np.array([right_corner0[:, 0], left_corner0[:, 0], right_corner1[:, 0], left_corner1[:, 0]]).flatten()
    y = np.array([right_corner0[:, 1], left_corner0[:, 1], right_corner1[:, 1], left_corner1[:, 1]]).flatten()

    triang = tri.Triangulation(x, y)
    refiner = tri.UniformTriRefiner(triang)
    return refiner.refine_triangulation(subdiv=refiner_sub_div)


def make_goal_(orange=False):
    goal_radius = 893

    goal = np.array([[- goal_radius, - common_values.BACK_WALL_Y],
                     [- goal_radius, - common_values.BACK_NET_Y],
                     [goal_radius, - common_values.BACK_WALL_Y],
                     [goal_radius, - common_values.BACK_NET_Y]])
    if orange:
        goal[:, 1] *= -1

    x = goal[:, 0].flatten()
    y = goal[:, 1].flatten()

    return tri.Triangulation(x, y)


_arena = make_arena_()
_blue_goal, _orange_goal = make_goal_(), make_goal_(True)
_boost_locations = np.array(common_values.BOOST_LOCATIONS)

arena_positions = np.stack([_arena.x, _arena.y], -1)
# We use a height dimension of 300 for the plots
arena_positions = np.append(arena_positions, np.ones((arena_positions.shape[0], 1)) * 300, -1)

_arena_positions_kdtree = KDTree(arena_positions)


def arena_contour(z: np.ndarray,
                  ball_position: np.ndarray = None,
                  ball_lin_vel: np.ndarray = None,
                  player_positions: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]] = None,
                  player_lin_vels: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]] = None,
                  goal_w=1,
                  player_idx: Union[int, None] = 0,
                  annotate_ball=False,
                  figsize: Union[int, Tuple[int, int]] = (12, 15),
                  ball_size=128,
                  player_size=128,
                  boost_pad_size=80,
                  contour_levels=80):
    """
    Contour plot of a reward function on the Rocket League arena

    :param z: Reward values for each point of the arena
    :param ball_position: Position of the ball, numpy array of shape (3,), optional
    :param ball_lin_vel: Linear velocity of the ball, numpy array of shape (3,), optional
    :param player_positions: Player positions in the arena. One of two options:
        i) numpy array of shape (n_players, 3), ii) 2-tuple of numpy arrays of shape (n_players, 3).
        Optional.
    :param player_lin_vels: Player linear velocity vectors, shapes similar to `player_positions`. Optional.
    :param goal_w: Goal reward, used for annotation only
    :param player_idx: The blue or orange team player index for which the rewards are plotted.
        If the player idx is between 0 and n_blue_team - 1 a blue team player is annotated.
        If the player idx is between n_blue_team and n_blue_team + n_orange_team - 1 an orange player is annotated.
        If the player idx is `None` no player is annotated.
    :param annotate_ball: Whether to annotate the ball with the reward
    :param figsize: The size of the figure. Can be either integer for a square plot or 2-tuple.
    :param ball_size: The ball marker size
    :param player_size: The player marker size
    :param boost_pad_size: The boost pad marker size
    :param contour_levels: Number of contour plot regions
    """

    if type(figsize) == int:
        figsize = (figsize, figsize)
    plt.figure(figsize=figsize)

    # --- Field plots ---

    # Arena plot
    arena_plot = plt.tricontourf(_arena, z, levels=contour_levels)

    # Goal plots
    orange_goal_z = np.ones_like(_orange_goal.y) * goal_w
    blue_goal_z = - np.ones_like(_blue_goal.y) * goal_w

    plt.tricontourf(_orange_goal, orange_goal_z, colors="tomato")
    plt.tricontourf(_blue_goal, blue_goal_z, colors="mediumturquoise")

    plt.text(_orange_goal.x.mean(), _orange_goal.y.mean(), str(goal_w), fontsize=14, ha="center", va="center")
    plt.text(_blue_goal.x.mean(), _blue_goal.y.mean(), str(-goal_w), fontsize=14, ha="center", va="center")

    # Boost pads
    small_boost_idx = _boost_locations[:, -1] == 70
    small_boost_pads = _boost_locations[small_boost_idx]
    large_boost_pads = _boost_locations[~small_boost_idx]
    plt.scatter(small_boost_pads[:, 0], small_boost_pads[:, 1],
                c="gold", marker="P", edgecolors="black", s=boost_pad_size, label="12% boost pads")
    plt.scatter(large_boost_pads[:, 0], large_boost_pads[:, 1],
                c="gold", marker="H", edgecolors="black", s=boost_pad_size * 1.5, label="100% boost pads")

    # --- Ball plots ---

    if ball_position is not None:
        plt.scatter(ball_position[0], ball_position[1],
                    c="red", marker="o", s=ball_size, label="Ball")
        if annotate_ball:
            # get the index of the nearest true point in the arena in order to retrieve the reward for that point
            _, idx = _arena_positions_kdtree.query(ball_position)
            ball_position = arena_positions[idx]
            plt.annotate(np.round(z[idx], 3), (ball_position[0], ball_position[1]))
        if ball_lin_vel is not None:
            # arena x axis is positive on the left-hand side, looking toward the orange goal
            # vectors cannot be inverted and are thus adjusted in the negative x direction
            plt.quiver(*ball_position[:2], -ball_lin_vel[0], ball_lin_vel[1],
                       color=['r'], headwidth=2.5, headlength=3, headaxislength=3)

    # --- Player plots ---

    if player_positions is not None:
        if type(player_positions) == tuple:
            # Blue team
            _, blue_idcs = _arena_positions_kdtree.query(player_positions[0])
            blue_team = arena_positions[blue_idcs]

            # Orange team
            _, orange_idcs = _arena_positions_kdtree.query(player_positions[1])
            orange_team = arena_positions[orange_idcs]
        else:
            _, blue_idcs = _arena_positions_kdtree.query(player_positions)
            blue_team = arena_positions[blue_idcs]
            orange_idcs, orange_team = None, None

        # Blue plot
        plt.scatter(blue_team[:, 0], blue_team[:, 1],
                    c="deepskyblue", marker="D", s=player_size, label="Blue team")
        # Orange plot
        if orange_team is not None:
            plt.scatter(orange_team[:, 0], orange_team[:, 1],
                        c="orangered", marker="D", s=player_size, label="Orange team")

        # Player annotation
        if player_idx is not None:
            if player_idx / len(blue_team) < 1:
                player_annot_rew_idx = blue_idcs[player_idx]
                player_annot_pos = blue_team[player_idx]
            else:
                player_annot_rew_idx = orange_idcs[player_idx - len(blue_team)]
                player_annot_pos = orange_team[player_idx - len(blue_team)]
            player_annot_rew = z[player_annot_rew_idx].round(3)
            plt.annotate(player_annot_rew, (player_annot_pos[0], player_annot_pos[1]))

        # TODO: add player forward vectors

        # TODO: add gradient vector from player position

        # Linear velocities
        if player_lin_vels is not None:
            if type(player_lin_vels) == tuple:
                blue_team_vels, orange_team_vels = player_lin_vels
                plt.quiver(*blue_team[:, :2].T, -blue_team_vels[:, 0], blue_team_vels[:, 1],
                           color=['deepskyblue'], headwidth=2.5, headlength=3, headaxislength=3)
                plt.quiver(*orange_team[:, :2].T, -orange_team_vels[:, 0], orange_team_vels[:, 1],
                           color=['orangered'], headwidth=2.5, headlength=3, headaxislength=3)
            else:
                plt.quiver(*blue_team[:, :2].T, -player_lin_vels[:, 0], player_lin_vels[:, 1],
                           color=['deepskyblue'], headwidth=2.5, headlength=3, headaxislength=3)

    # --- Final steps ---
    # Arena x axis is positive on the left-hand side, looking toward the orange goal
    plt.gca().invert_xaxis()
    plt.legend()
    plt.colorbar(arena_plot)
    plt.show()
