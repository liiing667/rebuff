from typing import Any, Dict, Optional


class RebuffError(ValueError):
    """Base error for Rebuff SDK failures.

    Inheriting from ``ValueError`` keeps older callers that caught
    ``ValueError`` working while still providing a named, importable error and
    structured evidence for security-sensitive failures.
    """

    def __init__(
        self, message: str, evidence: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.evidence: Dict[str, Any] = evidence or {}
        # A raised error never represents a safe result.  Expose the security
        # posture explicitly for callers that branch on error metadata.
        self.injection_detected = True
        self.detected = True

