from __future__ import annotations

"""Store module."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class ConsentRecord:
    """Data container for consentrecord information."""
    subject_id: str
    scope: str
    granted_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None

    @property
    def active(self) -> bool:
        """Active.

            Returns:
                Boolean result.
            """
        if self.revoked_at is not None:
            return False
        if (
            self.expires_at is not None
            and datetime.now(timezone.utc) >= self.expires_at
        ):
            return False
        return True


class ConsentStore:
    """In-memory consent store. Replace with DB in production."""

    def __init__(self) -> None:
        """Initialize instance.
            """
        self._by_subject: dict[str, list[ConsentRecord]] = {}

    def grant(
        """Grant.

            Args:
                subject_id: Subject id.
                scope: Scope.
                ttl_days: Ttl days.

            Returns:
                ConsentRecord instance.
            """
        self, subject_id: str, scope: str, ttl_days: int | None = None
    ) -> ConsentRecord:
        now = datetime.now(timezone.utc)
        exp = now + timedelta(days=int(ttl_days)) if ttl_days else None
        rec = ConsentRecord(
            subject_id=subject_id, scope=scope, granted_at=now, expires_at=exp
        )
        self._by_subject.setdefault(subject_id, []).append(rec)
        return rec

    def revoke(self, subject_id: str, scope: str) -> bool:
        """Revoke.

            Args:
                subject_id: Subject id.
                scope: Scope.

            Returns:
                Boolean result.
            """
        ok = False
        for rec in self._by_subject.get(subject_id, []):
            if rec.scope == scope and rec.active:
                rec.revoked_at = datetime.now(timezone.utc)
                ok = True
        return ok

    def has_consent(self, subject_id: str, scope: str) -> bool:
        """Check if has consent.

            Args:
                subject_id: Subject id.
                scope: Scope.

            Returns:
                True if condition is met, False otherwise.
            """
        return any(
            rec.scope == scope and rec.active
            for rec in self._by_subject.get(subject_id, [])
        )
