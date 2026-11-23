from typing import Optional


class RebuffError(Exception):
    """Base class for all Rebuff errors."""


class LeakLoggingError(RebuffError):
    """
    Raised when a canary leak was detected in the completion but the leak
    could not be durably logged/alerted.

    The canary word WAS present in the completion, so the leak itself is a
    fact. This error means the alert/log state could not be confirmed; the
    caller must treat the event as a confirmed leak and retry the logging
    with the same ``leak_id`` rather than assuming the alert was raised or
    that the system is in a safe state.
    """

    def __init__(self, message: str, leak_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.leak_id = leak_id


class VectorStoreInitializationError(RebuffError):
    """
    Raised when the vector store cannot be initialized.

    A vector check that cannot run must never be silently treated as "no
    injection detected" (a zero score). Failing closed keeps an
    uninitialized vector store from downgrading detection to a safe verdict.
    """
