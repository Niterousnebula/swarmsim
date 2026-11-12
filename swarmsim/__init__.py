"""swarmsim: modular drone-swarm simulation environment.

Four independent modules that talk through plain data (files / dataclasses):

    terrain  -> procedural city generator (OSM statistics + satellite palette)
    models   -> prebuilt vehicle meshes + URDFs + performance specs
    physics  -> PyBullet world: gravity, wind, aerodynamics, projectiles
    rl       -> PettingZoo environment, trainer and policy-runner plugin
"""
__version__ = "0.1.0"


def _import_pybullet_quietly():
    """PyBullet prints a "pybullet build time" banner from C when it is first
    imported. Import it here once with the process's output pointed at nowhere."""
    import os
    import sys
    if "pybullet" in sys.modules:
        return
    try:
        sys.stdout.flush(); sys.stderr.flush()
        saved = {fd: os.dup(fd) for fd in (1, 2)}
        sink = os.open(os.devnull, os.O_WRONLY)
    except (OSError, ValueError, AttributeError):
        return                                  # no real output streams (some embedded hosts): nothing to hide
    try:
        for fd in saved:
            os.dup2(sink, fd)
        import pybullet  # noqa: F401
        try:                                    # the banner may still sit in C's own buffer
            import ctypes
            (ctypes.CDLL("ucrtbase") if os.name == "nt" else ctypes.CDLL(None)).fflush(None)
        except Exception:
            pass
    except ImportError:
        pass                                    # reported properly by whoever needs it
    finally:
        for fd, copy in saved.items():
            os.dup2(copy, fd); os.close(copy)
        os.close(sink)


_import_pybullet_quietly()
