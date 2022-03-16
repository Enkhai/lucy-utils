import json
import os
from typing import Union

import numpy as np
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.math import cosine_similarity
from rlgym.utils.reward_functions import RewardFunction


class LucyReward(RewardFunction):
    """
    *Always make sure that this reward is a different copy for each match.*

    Reward inspired by Necto. Compatible with SB3CombinedLogRewardCallback.\n
    Suggested logback reward names:

    - Goal / Team goal / Concede
    - Shot
    - Save
    - Demolish / demolished
    - Touch aerial with toward-goal acceleration
    - Inversely utility-weighted save boost reward
    - Utility total
    - Ball2goal distance
    - Ball2goal velocity
    - Ball y axis
    - Player2ball velocity
    - Player2ball distance
    - Distance-weighted align ball2goal
    - Offensive potential

    Original Necto reward: https://github.com/Rolv-Arild/Necto/blob/master/training/reward.py
    """

    _goal_depth = common_values.BACK_NET_Y - common_values.BACK_WALL_Y + common_values.BALL_RADIUS
    _blue_objective = np.array(common_values.ORANGE_GOAL_BACK)
    _orange_objective = np.array(common_values.BLUE_GOAL_BACK)

    def __init__(self,
                 # --- event weights ---
                 goal_w=10,
                 shot_w=1,
                 save_w=3,
                 demo_w=2,
                 touch_height_w=1,
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
                 # --- log ---
                 n_players=6,
                 file_location: str = 'combinedlogfiles'
                 ):
        super(LucyReward, self).__init__()

        # 6 event rewards, 1 utility total, 3 state utilities, 4 player utilities
        self.n_returns = 6 + 1 + 3 + 4
        self.n_players = n_players
        self.returns: Union[np.ndarray, None] = None
        # Make log for TensorBoard reward logging
        self._init_log(file_location)

        # Maintain states to check whether the time frame has changed in order to
        # compute complete rewards and utilities for all n players in the time frame
        self.current_state: Union[GameState, None] = None
        self.last_state: Union[GameState, None] = None
        self.n = 0  # player reward counter
        self.player_rewards: Union[np.ndarray, None] = None
        self.state_utilities: Union[np.ndarray, None] = None
        self.player_utilities: Union[np.ndarray, None] = None

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

        # Compute utility expected and maximum value
        self._compute_utility_expected_max()

    def _compute_utility_expected_max(self):
        # Multiplied by min values
        utility_min = (self.ball2goal_dist_w * -0.5 +
                       self.ball2goal_vel_w * -1 +
                       self.ball_y_w * -1 +
                       self.player2ball_vel_w * -1 +
                       self.player2ball_dist_w * 0 +
                       self.align_ball_w * -1 +
                       self.offensive_potential_w * -1)
        # Multiplied by max values
        utility_max = (self.ball2goal_dist_w * 1 +
                       self.ball2goal_vel_w * 1 +
                       self.ball_y_w * 1 +
                       self.player2ball_vel_w * 1 +
                       self.player2ball_dist_w * 1 +
                       self.align_ball_w * 1 +
                       self.offensive_potential_w * 1)
        self.utility_expected = (utility_min + utility_max) / 2
        self.utility_max = utility_max - self.utility_expected  # adjust by mean

    def _init_log(self, file_location):
        """
        SB3CombinedLogReward init logic
        """

        # Make sure there is a folder to dump to
        os.makedirs(file_location, exist_ok=True)
        self.file_location = f'{file_location}/rewards.txt'
        self.lockfile = f'{file_location}/reward_lock'

        # Initialize the array that will store the episode totals
        self.returns = np.zeros((self.n_players, self.n_returns))

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
        with open(self.file_location, 'w'):
            pass

        # Release the lock
        try:
            os.remove(self.lockfile)
        except FileNotFoundError:
            print('No lock to release! ')

    def reset(self, initial_state: GameState):
        self.returns = np.zeros((self.n_players, self.n_returns))

        # following Necto reward logic
        self.n = 0
        self.last_state = None
        self.current_state = initial_state
        self.player_rewards = None
        self.player_utilities = self._compute_utilities()

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # following Necto reward logic
        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state
            self._compute_rewards()
            self.n = 0
        rew = self.player_rewards[self.n].sum()
        self.n += 1
        return float(rew)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # following Necto reward logic
        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state
            self._compute_rewards()
            self.n = 0
        rew = self.player_rewards[self.n].sum()
        self.n += 1
        self._log_returns()

        return float(rew)

    def _log_returns(self):
        """
        Logs returns to be read from an SB3CombinedLogRewardCallback. Follows SB3CombinedLogReward logging logic.
        """
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
                print(f'Error obtaining lock in SB3CombinedLogReward.get_final_reward:\n{e}')

        # Write the rewards to file and reset
        with open(self.file_location, 'a') as f:
            for p_ret in self.returns:
                f.write('\n' + json.dumps(p_ret.tolist()))

        # Release the lock
        try:
            os.remove(self.lockfile)
        except FileNotFoundError:
            print('No lock to release! ')

    def _compute_utilities(self):
        """
        Computes and returns utilities. Always update the current state before calling the method.
        """
        ball_pos = self.current_state.ball.position
        ball_lin_vel = self.current_state.ball.linear_velocity

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

            state_utilities.append([ball2goal_dist * self.ball2goal_dist_w,
                                    ball2goal_vel * self.ball2goal_vel_w,
                                    ball_y * self.ball_y_w])
        # 2 * 3
        state_utilities = np.array(state_utilities)

        # Player utilities: player to ball velocity, player to ball distance,
        # distance-weighted align ball to goal, offensive potential
        player_utilities = []
        for p in self.current_state.players:
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

            player_utilities.append([player2ball_vel * self.player2ball_vel_w,
                                     player2ball_dist * self.player2ball_dist_w,
                                     align_ball * self.align_ball_w,
                                     offensive_potential * self.offensive_potential_w])
        # n_players * 4
        player_utilities = np.array(player_utilities)

        return state_utilities, player_utilities

    def _compute_rewards(self):
        """
        Computes and updates rewards and utilities. Always update the current and last state before calling the method.
        """
        state_utilities, player_utilities = self._compute_utilities()

        utility_sum = np.zeros(self.n_players)
        utility_sum[:self.n_players / 2] = player_utilities[:self.n_players / 2].sum(0) + state_utilities[0].sum()
        utility_sum[self.n_players / 2:] = player_utilities[self.n_players / 2:].sum(0) + state_utilities[1].sum()

        player_rewards = []

        # Goal and team goal related data
        d_orange = self.current_state.orange_score - self.last_state.orange_score
        d_blue = self.current_state.blue_score - self.last_state.blue_score
        goal_scored = [False, False]

        for p in self.current_state.players:
            last_player_state: PlayerData
            last_player_state = self.last_state.players[p.car_id]

            # Event rewards
            # Scoring related: goal / team goal / concede, shot, save
            # Goal
            goal = 0
            if p.match_goals > last_player_state.match_goals:
                goal = self.goal_w
                goal_scored[p.team_num] = True
            # Concede
            if p.team_num == common_values.BLUE_TEAM:
                if d_orange > 0:
                    goal = -self.goal_w
            else:
                if d_blue > 0:
                    goal = -self.goal_w
            # Shot
            shot = 0
            if p.match_shots > last_player_state.match_shots:
                shot = self.shot_w
            # Save
            save = 0
            if p.match_saves > last_player_state.match_saves:
                save = self.save_w

            # Others: demolish / demolished, touch aerial with acceleration toward goal,
            # utility-weighted save boost reward
            # Demolish / demolished
            demo = 0
            if p.is_demoed and not last_player_state.is_demoed:
                demo -= self.demo_w
            if p.match_demolishes > last_player_state.match_demolishes:
                demo += self.demo_w

            # Touch aerial with acceleration toward goal
            touch_aerial_accel = 0
            if p.ball_touched:
                ball_pos = self.current_state.ball.position
                ball_vel = self.current_state.ball.linear_velocity
                last_ball_vel = self.last_state.ball.linear_velocity
                last_ball_pos = self.last_state.ball.position

                if p.team_num == common_values.BLUE_TEAM:
                    team_objective = self._blue_objective
                else:
                    team_objective = self._orange_objective

                ball2goal_pos_diff = ball_pos - team_objective
                # we don't normalize to get the true velocity
                ball2goal_vel = np.dot(ball2goal_pos_diff / np.linalg.norm(ball2goal_pos_diff),
                                       ball_vel)
                last_ball2goal_pos_diff = last_ball_pos - team_objective
                last_ball2goal_vel = np.dot(last_ball2goal_pos_diff / np.linalg.norm(last_ball2goal_pos_diff),
                                            last_ball_vel)

                touch_aerial_accel = (self.touch_height_w * ball_pos[2] / 2250 +  # normalize by max height
                                      # we reward velocity change with 1 if it changes by 2300,
                                      # aimed toward the opponent goal
                                      (self.touch_accel_w * (ball2goal_vel - last_ball2goal_vel) /
                                       common_values.CAR_MAX_SPEED)
                                      )

            # Utility-weighted save boost reward
            boost_diff = np.sqrt(p.boost_amount) - np.sqrt(p.boost_amount)
            if boost_diff >= 0:
                save_boost = boost_diff
            else:
                # we penalize boost loss less for states that have very large or very small utility
                save_boost = boost_diff * ((self.utility_max - np.abs(utility_sum[p.car_id])) / self.utility_max)

            player_rewards.append([goal, shot, save, demo, touch_aerial_accel, save_boost])

        # Team goal
        # If there was no scorer, reward with utility / 4
        team_goal = np.zeros(self.n_players)
        if not goal_scored[0] and d_blue > 0:
            team_goal[:self.n_players / 2] = (4 * (np.abs(utility_sum[:self.n_players / 2]) - self.utility_expected) /
                                              self.utility_max)
        if not goal_scored[1] and d_orange > 0:
            team_goal[self.n_players / 2:] = (4 * (np.abs(utility_sum[self.n_players / 2:]) - self.utility_expected) /
                                              self.utility_max)

        self.player_rewards = np.concatenate((np.array(player_rewards), utility_sum), axis=-1)

        self.returns += np.concatenate((self.player_rewards, state_utilities, player_utilities))
