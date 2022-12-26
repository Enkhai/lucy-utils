from .element import ElementActor
from .necto import NectoActor


def _get_path():
    from pathlib import Path

    return str(Path(__file__).parent.resolve())


_path = _get_path()
