
"""."""

import cProfile


def start_profiling():
    """Start code execution profile."""
    p = cProfile.Profile()
    p.enable()
    return p
