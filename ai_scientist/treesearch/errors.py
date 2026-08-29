"""Shared fail-closed errors for bounded research execution."""


class ExperimentCannotContinueError(RuntimeError):
    """Raised when a run has no scientifically valid state to continue from."""


class MultiSeedGateRejectedError(ExperimentCannotContinueError):
    """Raised after a complete seed batch refutes a candidate's promotion."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(f"Multi-seed candidate rejected by {reason_code}")


__all__ = ["ExperimentCannotContinueError", "MultiSeedGateRejectedError"]
