import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class OnStepRewardLogCallback(BaseCallback):
    """
    Logback useful for reading reward logs from a constantly extending dumpfile
    """
    def __init__(self, log_dumpfile: str,
                 reward_names_: list,
                 verbose=0):
        super(OnStepRewardLogCallback, self).__init__(verbose)
        self.log_dumpfile = Path(log_dumpfile)
        self.log_dumpfile_io = None
        self.reward_names = reward_names_

    def _on_step(self) -> bool:
        if not self.log_dumpfile_io and self.log_dumpfile.exists():
            self.log_dumpfile_io = open(self.log_dumpfile, "r")

        if self.log_dumpfile_io:
            line = self.log_dumpfile_io.readline()
            if line and line != "\n":
                rewards = json.loads(line)
                for i in range(len(rewards)):
                    self.model.logger.record(key="rewards/" + self.reward_names[i], value=rewards[i])
        return True

    def _on_training_end(self) -> None:
        self.log_dumpfile_io.close()
