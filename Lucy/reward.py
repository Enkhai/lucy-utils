import os

import numpy as np
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.math import cosine_similarity


class LucyReward(RewardFunction):
    _goal_depth = common_values.BACK_NET_Y - common_values.BACK_WALL_Y + common_values.BALL_RADIUS
    _blue_objective = np.array(common_values.ORANGE_GOAL_BACK)
    _orange_objective = np.array(common_values.BLUE_GOAL_BACK)

    def __init__(self,
                 # --- event weights ---
                 goal_w=10,
                 shot_w=1,
                 save_w=3,
                 demo_w=2,
                 touch_height_w=1,  # TODO: calibrate touch aerial reward and weight
                 touch_accel_w=0.25,
                 # --- utility weights ---
                 # state
                 ball2goal_dist_w=4,
                 ball2goal_vel_w=1.5,
                 ball_y_w=1,
                 # player
                 player2ball_vel_w=0.25,
                 player2ball_dist_w=0.5,
                 align_ball_w=0.75,
                 offensive_potential_w=4,
                 # log
                 file_location: str = 'combinedlogfiles'
                 ):
        super(LucyReward, self).__init__()

        # Make log for TensorBoard reward logging
        self._init_log(file_location)

        # Maintain states to check whether the time frame has changed in order to
        # compute complete rewards and utilities for all n players in the time frame
        self.current_state = None
        self.last_state = None
        # player reward counter
        self.n = 0
        self.rewards = None
        self.player_utilities = None

        # Event reward weights (original reward R)
        # scoring related
        self.goal_w = goal_w
        self.shot_w = shot_w
        self.save_w = save_w
        # others
        self.demo_w = demo_w
        self.touch_height_w = touch_height_w
        self.touch_accel_w = touch_accel_w

        # Utility weights (shaping reward F)
        # state
        self.ball2goal_dist_w = ball2goal_dist_w
        self.ball2goal_vel_w = ball2goal_vel_w
        self.ball_y_w = ball_y_w
        # player
        self.player2ball_vel_w = player2ball_vel_w
        self.player2ball_dist_w = player2ball_dist_w
        self.align_ball_w = align_ball_w
        self.offensive_potential_w = offensive_potential_w

    def _init_log(self, file_location):
        """
        SB3CombinedLogReward init logic
        """

        # Make sure there is a folder to dump to
        os.makedirs(file_location, exist_ok=True)
        self.file_location = f'{file_location}/rewards.txt'
        self.lockfile = f'{file_location}/reward_lock'

        # Initializes the array that will store the episode totals
        # 6 event rewards, 3 state utilities, 4 player utilities, 1 utility total
        self.n_scores = 6 + 3 + 4 + 1
        self.scores = np.zeros(self.n_scores)

        # Obtain the lock
        while True:
            try:
                open(self.lockfile, 'x')
                break
            except FileExistsError:
                pass
            except PermissionError:
                pass
            except Exception as e:
                print(f'Error obtaining lock in SB3CombinedLogReward.__init__:\n{e}')

        # Empty the file by opening in w mode
        with open(self.file_location, 'w') as f:
            pass

        # Release the lock
        try:
            os.remove(self.lockfile)
        except FileNotFoundError:
            print('No lock to release! ')

    def reset(self, initial_state: GameState):
        self.scores = np.zeros(self.n_scores)

        # following Necto reward logic
        self.n = 0
        self.last_state = None
        self.rewards = None
        self.current_state = initial_state
        self.player_utilities = self._compute_utilities(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # following Necto reward logic
        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state
            self._compute_rewards(state)
            self.n = 0
        rew = self.rewards[self.n]
        self.n += 1
        return float(rew)

    def _compute_utilities(self, state):
        ball_pos = state.ball.position
        ball_lin_vel = state.ball.linear_velocity

        # State utilities: ball to goal distance, ball to goal velocity, ball y axis position
        # state utility is non-symmetric for the two teams, we must compute them separately
        state_utilities = []
        for objective, team_view in zip((self._blue_objective, self._orange_objective), (1, -1)):
            ball2goal_pos_diff = ball_pos - objective
            ball2goal_pos_diff_norm = np.linalg.norm(ball2goal_pos_diff)

            # Ball to goal distance
            ball2goal_dist = np.exp(-0.5 * (ball2goal_pos_diff_norm - self._goal_depth) /
                                    (4570 * 1.1))  # dispersion factor 1.1
            ball2goal_dist = (ball2goal_dist - 0.5) * 2  # signed, no density
            # Ball to goal velocity
            ball2goal_vel = np.dot(ball2goal_pos_diff / ball2goal_pos_diff_norm,
                                   ball_lin_vel / common_values.BALL_MAX_SPEED)
            # Ball y axis
            ball_y = team_view * ball_pos[1] / (common_values.BACK_WALL_Y + common_values.BALL_RADIUS)

            # TODO: log reward and utility scores
            state_utilities.append(ball2goal_dist * self.ball2goal_dist_w +
                                   ball2goal_vel * self.ball2goal_vel_w +
                                   ball_y * self.ball_y_w)
        state_utilities = np.array(state_utilities)

        # Player utilities: player to ball velocity, player to ball distance,
        # distance-weighted align ball to goal, offensive potential
        player_utilities = []
        for p in state.players:
            player_pos = p.car_data.position
            player_lin_vel = p.car_data.linear_velocity
            if p.team_num == common_values.BLUE_TEAM:
                team_objective = self._blue_objective
                opp_objective = self._orange_objective
            else:
                team_objective = self._orange_objective
                opp_objective = self._blue_objective

            player2ball_pos_diff = ball_pos - player_pos
            player2ball_pos_diff_norm = np.linalg.norm(player2ball_pos_diff)

            # Player to ball velocity
            player2ball_vel = np.dot(player2ball_pos_diff / player2ball_pos_diff_norm,
                                     player_lin_vel / common_values.CAR_MAX_SPEED)

            # Player to ball distance
            player2ball_dist = np.exp(-0.5 * (player2ball_pos_diff_norm - common_values.BALL_RADIUS) /
                                      common_values.CAR_MAX_SPEED)

            # * Align ball to goal (auxiliary)
            # we use equal offensive and defensive alignment weights of 0.5
            regular_align_ball = (
                    0.5 * cosine_similarity(player2ball_pos_diff, player_pos - team_objective) +  # offense
                    0.5 * cosine_similarity(player2ball_pos_diff, opp_objective - player_pos)  # defense
            )

            # Distance-weighted align ball to goal
            align_ball = regular_align_ball * player2ball_dist
            # square root because we multiply two values between -1 and 1
            # "weighted" product (n_1 * n_2 * ... * n_N) ^ (1 / N)
            align_ball = np.sqrt(np.abs(align_ball)) * np.sign(align_ball)

            # Offensive potential
            # logical AND
            # when both alignment and player to ball velocity are negative we must get a negative output
            # player2ball_dist is positive only, no need to compute for sign
            offensive_potential_sign = (((player2ball_vel >= 0) and (regular_align_ball >= 0)) - 0.5) * 2
            # cube root because we multiply three values between -1 and 1
            # "weighted" product (n_1 * n_2 * ... * n_N) ^ (1 / N)
            offensive_potential = ((np.abs(regular_align_ball * player2ball_vel * player2ball_dist) ** (1 / 3)) *
                                   offensive_potential_sign)

            # TODO: log reward and utility scores
            player_utilities.append(player2ball_vel * self.player2ball_vel_w +
                                    player2ball_dist * self.player2ball_dist_w +
                                    align_ball * self.align_ball_w +
                                    offensive_potential * self.offensive_potential_w)
        player_utilities = np.array(player_utilities)

        return state_utilities, player_utilities

    def _compute_rewards(self, state):
        # TODO: fill this
        #  continue from here https://github.com/Rolv-Arild/Necto/blob/master/training/reward.py
        # thought, inspired from Necto: if there is no scorer, for various reasons, attribute some points to team

        # TODO: log reward and utility scores

        pass

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # following Necto reward logic
        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state
            self._compute_rewards(state)
            self.n = 0
        rew = self.rewards[self.n]
        self.n += 1

        # TODO: write reward and utility scores
        self.scores += [0]

        return float(rew)
