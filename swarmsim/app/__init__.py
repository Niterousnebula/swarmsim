"""Control app: a local web page with a live 3-D view of the simulation, camera
and piloting controls, and buttons to fly models and start training.

    python main.py            (or: python -m swarmsim.app)
"""
from .server import serve

__all__ = ["serve"]
