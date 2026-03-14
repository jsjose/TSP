"""
LKH-3 solver wrapper.

By default the solver looks for an ``LKH`` binary on the system PATH.
If the binary lives in a non-standard location, configure it once at
the module level before instantiating any solver:

    import algorithms.lkh as lkh_mod
    lkh_mod.configure(lkh_binary_path="/opt/LKH-3/LKH")

Or pass the path directly to each instance:

    solver = LKH3Solver(dist_matrix, lkh_path="/opt/LKH-3/LKH")
"""

import os
import shutil
import tempfile
import time

import numpy as np

try:
    import lkh as _lkh_module
except ImportError:
    _lkh_module = None

# Module-level override for the LKH binary path.
# None  → resolve via PATH (shutil.which("LKH"))
# str   → absolute or relative path to the LKH binary
_LKH_BINARY_PATH: str | None = None


def configure(lkh_binary_path: str) -> None:
    """Set the LKH binary path for all subsequent solver instances.

    Call this once at startup when the LKH binary is not on PATH.

    Args:
        lkh_binary_path: Absolute or relative path to the LKH executable,
            e.g. ``"/opt/LKH-3/LKH"`` or ``"./LKH"``.

    Example:
        >>> import algorithms.lkh as lkh_mod
        >>> lkh_mod.configure("/opt/LKH-3/LKH")
        >>> solver = lkh_mod.LKH3Solver(dist_matrix)
    """
    global _LKH_BINARY_PATH
    _LKH_BINARY_PATH = lkh_binary_path


def _resolve_binary(lkh_path: str | None) -> str | None:
    """Return the resolved LKH binary path, or None if not found.

    Priority order:
    1. ``lkh_path`` argument (instance-level override)
    2. Module-level ``_LKH_BINARY_PATH`` (set via ``configure()``)
    3. ``LKH`` on the system PATH (``shutil.which``)

    Args:
        lkh_path: Instance-level path override, or None.

    Returns:
        Resolved path string if the binary exists, else None.
    """
    candidate = lkh_path or _LKH_BINARY_PATH
    if candidate is not None:
        return candidate if os.path.isfile(candidate) else None
    return shutil.which("LKH")


class LKH3Solver:
    """Wraps the LKH-3 heuristic via the ``lkh`` Python bridge package.

    Args:
        dist_matrix: Square distance matrix (NxN).
        lkh_path: Optional path to the LKH binary. Overrides both the
            module-level default and the system PATH for this instance.
    """

    @staticmethod
    def is_available(lkh_path: str | None = None) -> bool:
        """Return True if the ``lkh`` package and the LKH binary are both found.

        Args:
            lkh_path: Optional explicit path to the LKH binary. Falls back to
                the module-level default and then to PATH discovery.

        Returns:
            True when both the Python package and the binary are present.
        """
        return _lkh_module is not None and _resolve_binary(lkh_path) is not None

    def __init__(self, dist_matrix: np.ndarray, lkh_path: str | None = None) -> None:
        self.B = np.array(dist_matrix)
        self.n = len(dist_matrix)
        self._binary = _resolve_binary(lkh_path)

    def _write_tsp_file(self, path: str) -> None:
        """Write a TSPLIB EXPLICIT FULL_MATRIX problem file."""
        with open(path, "w") as f:
            f.write("NAME: TSP\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {self.n}\n")
            f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in self.B.astype(int).tolist():
                f.write(" ".join(str(v) for v in row) + "\n")
            f.write("EOF\n")

    def solve(self, runs: int = 10, max_trials: int = 1000) -> tuple[list[int] | None, int | None]:
        """Solve TSP using LKH-3.

        Args:
            runs: Number of independent LKH runs (higher → better quality).
            max_trials: Maximum number of trials per run.

        Returns:
            ``(path, cost)`` where path is a list of 0-based city indices
            forming a closed tour, or ``(None, None)`` on failure.

        Raises:
            RuntimeError: If the LKH binary could not be found.
        """
        if self._binary is None:
            raise RuntimeError(
                "LKH binary not found. Install LKH-3 and either add it to "
                "PATH, call algorithms.lkh.configure('/path/to/LKH'), or pass "
                "lkh_path= to the constructor."
            )

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tsp")
        os.close(tmp_fd)
        self._write_tsp_file(tmp_path)

        start_time = time.time()
        try:
            solution = _lkh_module.solve(
                solver=self._binary,
                problem_file=tmp_path,
                runs=runs,
                max_trials=max_trials,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        _ = time.time() - start_time  # available if callers need it

        if not solution:
            return None, None

        # Handle both list-of-nodes and list-of-routes return formats
        path_indices = solution[0] if isinstance(solution[0], list) else solution

        # LKH uses 1-based indexing → convert to 0-based
        path = [node - 1 for node in path_indices]
        path.append(path[0])  # close the tour

        cost = int(sum(self.B[path[i]][path[i + 1]] for i in range(self.n)))
        return path, cost
