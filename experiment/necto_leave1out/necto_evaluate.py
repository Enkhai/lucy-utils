from rlgym.utils.state_setters import DefaultState

from obs import NectoObs
from reward import NectoRewardFunction
from experiment.lucy_match_params import LucyTerminalConditions, LucyAction
from utils.load_evaluate import load_and_evaluate

if __name__ == '__main__':
    load_and_evaluate("../../models/NectoTest_Perceiver/model_417280000_steps.zip",
                      2,
                      LucyTerminalConditions(15),
                      NectoObs(),
                      DefaultState(),  # we use the default state for evaluation
                      LucyAction(),
                      NectoRewardFunction(),
                      )
