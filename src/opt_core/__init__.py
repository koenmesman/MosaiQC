"""Compatibility shim for the old ``opt_core`` import path.

New code should import :mod:`MosaiQC`. The old ``opt_core._core`` name is kept
as an alias for :mod:`MosaiQC.native`.
"""

from MosaiQC import *  # noqa: F401,F403
from MosaiQC import native as _core
