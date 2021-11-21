# RLGym installation

## Requirements

- Windows 10
- Rocket League, either:
    - [Steam version](https://store.steampowered.com/app/252950/Rocket_League/)
    - [Epic Games version](https://www.epicgames.com/store/en-US/p/rocket-league) <br>

  For either, you should have the respective client installed on your PC and an account
    - [Epic Games launcher](https://www.epicgames.com/store/en-US/download)
    - [Steam](https://store.steampowered.com/about/)
- [Bakkesmod](https://www.bakkesmod.com/): <br>
  Bakkesmod is a mod for Rocket League that injects into the game and can help you become better at it by helping you
  train and master various game moves.
  For RLGym, it is used for providing to and retrieving feedback from the game, allowing a reinforcement learning agent
  to interact with it and retrieve the game state.<br>
  Bakkesmod also offers a variety of [plugins](https://bakkesplugins.com/). <br>
  To install, follow the instructions in the [download page](https://www.bakkesmod.com/download.php). For RLGym, you
  simply need to install and update, and make sure it is running for the training process to work.<br>
  For antivirus issues consult
  [this](https://docs.google.com/spreadsheets/d/e/2PACX-1vSLd3OucDGczgvFDa_D4I72MYNVhskJMe-pA8Bi5eFBuCADixLR1QleIE-X8eE_4L-AlLNhIm6A7fTK/pubhtml)
  .
- The RLGym Bakkesmod plugin. This is installed (nearly) automatically when installing `rlgym` through pip. We will go
  through this in the installation process later on.
- An environment with Python >= 3.7.

## Installation (Anaconda)

1. Create/have an existing environment with Python 3.7 or later
    - Environment creation: `conda create -n rlgym python=3.8`
    - Activate environment: `conda activate rlgym`
2. Make sure the following packages are installed:<br>
   `pip install six pywin32 numpy comtypes cloudpickle pywinauto psutil gym`<br>
   This is necessary in order for the Bakkesmod RLGym plugin to be installed because some of these packages are used in
   the installation process, which unfortunately takes place before the installation of packages required for `rlgym`.
   <br>
   The relative error produced can be found [here](https://github.com/pypa/pip/issues/8368).
3. Install RLGym via pip: `pip install rlgym`
4. If you are having problems with RLGym imports/setting up the game environment/etc. you can:
    - copy the `pythoncom37.dll` and `pywintypes37.dll` files found in
      your `<Anaconda installation folder>\Lig\pywin32_system32` to `C:\Windows\System32`
    - run `python .\pywin32_postinstall.py -install`. The script was taken
      from [here](https://github.com/mhammond/pywin32/blob/main/pywin32_postinstall.py).

   These are used for fixing `win32pipe` and `win32file` imports used by the `rlgym` module. Either may probably be
   fine.
5. Usually, the PyTorch CPU version is installed by Stable Baselines (installed next). To install the GPU version, given
   that you have a card that supports CUDA 10.2, [run](https://pytorch.org/get-started/locally/): <br>
   `pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html`
6. Install Stable Baselines 3:
    - With additional dependencies like Tensorboard, OpenCV, `atari-py`: `pip install stable-baselines3[extra]`
    - Without additional dependencies: `pip install stable-baselines3`
7. You can also install `rlgym-tools` for additional environment alternatives, replay to `GameState`converter, and
   advanced reward functions.
