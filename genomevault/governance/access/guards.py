"""Guards module."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from genomevault.governance.consent.store import ConsentStore

_CONSENT = ConsentStore()  # simple singleton for app process


def get_consent_store() -> ConsentStore:
    """Get consent store.
    Returns:
        ConsentStore"""
    return _CONSENT


def require_consent(scope: str):
    """Retrieve consent store.

    Returns:
        The consent store.
    """
    """Require consent.

        Args:
            scope: Scope.

        Returns:
            Operation result.

        Raises:
            HTTPException: When operation fails.
        """

    def dep(
        subject_id: str | None = Header(default=None, alias="X-Subject-ID"),
        store: ConsentStore = Depends(get_consent_store),
    ):
        """Dep.

        Args:
            subject_id: Subject id.
            store: Store.

        Returns:
            Operation result.

        Raises:
            HTTPException: When operation fails.
        """
        if subject_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Missing X-Subject-ID"
            )
        if not store.has_consent(subject_id, scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No consent for scope '{scope}'",
            )
        return subject_id

    return dep
