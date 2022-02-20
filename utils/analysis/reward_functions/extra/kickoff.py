from utils.analysis.reward_functions.common.player_ball import velocity_player2ball


def kickoff(ball_position, player_position, player_lin_velocity, use_scalar_projection=False):
    if ball_position[0] == 0 and ball_position[1] == 0:
        return velocity_player2ball(player_position,
                                    player_lin_velocity,
                                    ball_position,
                                    use_scalar_projection)
    return 0
