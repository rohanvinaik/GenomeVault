"""File upload handling middleware."""

from __future__ import annotations

from fastapi import UploadFile, HTTPException

MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB
ALLOWED_EXTENSIONS = {".vcf", ".vcf.gz", ".fastq", ".fastq.gz", ".fq", ".fq.gz", ".bam", ".sam"}


async def validate_upload_file(file: UploadFile) -> None:
    """
    Validate uploaded file.

    Args:
        file: The uploaded file to validate

    Raises:
        HTTPException: If file is invalid (wrong type or too large)
    """
    # Check extension
    if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check file size (read in chunks to avoid memory issues)
    chunk_size = 1024 * 1024  # 1 MB
    total_size = 0

    while chunk := await file.read(chunk_size):
        total_size += len(chunk)
        if total_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024**3)} GB"
            )

    # Reset file pointer
    await file.seek(0)
