import numpy as np
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction


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
                 offensive_potential_w=4,
                 ball2goal_dist_w=4,
                 ball2goal_vel_w=1.5,
                 ball_y_w=1,
                 align_ball_w=0.75,
                 player2ball_dist_w=0.5,
                 player2ball_vel_w=0.25
                 ):
        super(LucyReward, self).__init__()
        # Maintain states to compute complete rewards and utilities for all n players in a time frame
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
        # main
        self.offensive_potential_w = offensive_potential_w
        self.ball2goal_dist_w = ball2goal_dist_w
        self.ball2goal_vel_w = ball2goal_vel_w
        self.ball_y_w = ball_y_w
        # others
        self.align_ball_w = align_ball_w
        self.player2ball_dist_w = player2ball_dist_w
        self.player2ball_vel_w = player2ball_vel_w

    def reset(self, initial_state: GameState):
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

        # `State` utility: ball to goal distance, ball to goal velocity, ball y axis position
        # `Player` utility: offensive potential, distance-weighted align ball to goal,
        # player to ball distance, player to ball velocity
        utilities = []
        for p in state.players:
            p_pos = p.car_data.position
            if p.team_num == common_values.BLUE_TEAM:
                team_objective = self._blue_objective
                opp_objective = self._orange_objective
            else:
                team_objective = self._orange_objective
                opp_objective = self._blue_objective

            ball2goal_pos_diff = ball_pos - team_objective
            ball2goal_pos_norm = np.linalg.norm(ball2goal_pos_diff)
            # Ball to goal distance
            ball2goal_dist = ball2goal_pos_norm - self._goal_depth
            ball2goal_dist = np.exp(-0.5 * ball2goal_dist / (4570 * 1.1))  # dispersion factor 1.1
            ball2goal_dist = (ball2goal_dist - 0.5) * 2  # signed, no density
            # Ball to goal velocity
            ball2goal_vel = np.dot(ball2goal_pos_diff / ball2goal_pos_norm,
                                   ball_lin_vel / common_values.BALL_MAX_SPEED)
            # Ball y axis
            ball_y = ball_pos[1] / (common_values.BACK_WALL_Y + common_values.BALL_RADIUS)
            if p.team_num:
                ball_y *= -1
            # TODO: complete the rest, based on reward building concepts
            # Player to ball velocity
            # Player to ball distance
            # * Align ball to goal (auxiliary)
            # Distance-weighted align ball to goal
            # Offensive potential

            utilities.append(0)  # TODO: fill this with final score

        return np.array(utilities)

    def _compute_rewards(self, state):
        # TODO: fill this
        #  continue from here https://github.com/Rolv-Arild/Necto/blob/master/training/reward.py
        # thought, inspired from Necto: if there is no scorer, for various reasons, attribute some points to team
        pass
