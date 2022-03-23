from rlgym.utils.state_setters import DefaultState

from experiment.lucy_match_params import LucyReward, LucyTerminalConditions, LucyObs, LucyAction
from rlgym_utils.load_evaluate import load_and_evaluate

if __name__ == '__main__':
    # load and evaluate computes mean reward and reward std for both teams
    load_and_evaluate("../models/Perceiver/model_449280000_steps.zip",
                      2,
                      LucyTerminalConditions(15),
                      LucyObs(),
                      DefaultState(),  # we use the default state for evaluation
                      LucyAction(),
                      LucyReward()
                      )
