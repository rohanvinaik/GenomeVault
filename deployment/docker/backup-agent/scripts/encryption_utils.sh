#!/bin/bash
# Encryption utilities for backup operations

set -euo pipefail

# Encryption configuration
ENCRYPTION_ALGORITHM="${ENCRYPTION_ALGORITHM:-aes-256-cbc}"
KEY_FILE="${ENCRYPTION_KEY_FILE:-/etc/encryption/backup-key}"
GPG_RECIPIENT="${GPG_RECIPIENT:-genomevault-backup@example.com}"

# Generate encryption key if not exists
generate_key() {
    if [[ ! -f "$KEY_FILE" ]]; then
        echo "Generating new encryption key..."
        openssl rand -base64 32 > "$KEY_FILE"
        chmod 600 "$KEY_FILE"
        echo "Encryption key generated at $KEY_FILE"
    fi
}

# Encrypt file using OpenSSL
encrypt_file() {
    local input_file="$1"
    local output_file="${2:-${input_file}.enc}"

    if [[ ! -f "$KEY_FILE" ]]; then
        echo "ERROR: Encryption key not found at $KEY_FILE"
        return 1
    fi

    openssl enc "-${ENCRYPTION_ALGORITHM}" \
        -salt \
        -in "$input_file" \
        -out "$output_file" \
        -pass "file:${KEY_FILE}"

    echo "File encrypted: $output_file"
}

# Decrypt file using OpenSSL
decrypt_file() {
    local input_file="$1"
    local output_file="${2:-${input_file%.enc}}"

    if [[ ! -f "$KEY_FILE" ]]; then
        echo "ERROR: Encryption key not found at $KEY_FILE"
        return 1
    fi

    openssl enc "-${ENCRYPTION_ALGORITHM}" \
        -d \
        -in "$input_file" \
        -out "$output_file" \
        -pass "file:${KEY_FILE}"

    echo "File decrypted: $output_file"
}

# Encrypt using GPG (alternative method)
gpg_encrypt() {
    local input_file="$1"
    local output_file="${2:-${input_file}.gpg}"

    gpg --encrypt \
        --recipient "$GPG_RECIPIENT" \
        --cipher-algo AES256 \
        --armor \
        --output "$output_file" \
        "$input_file"

    echo "File GPG encrypted: $output_file"
}

# Decrypt using GPG
gpg_decrypt() {
    local input_file="$1"
    local output_file="${2:-${input_file%.gpg}}"

    gpg --decrypt \
        --output "$output_file" \
        "$input_file"

    echo "File GPG decrypted: $output_file"
}

# Calculate and store checksum
calculate_checksum() {
    local file="$1"
    local checksum_file="${file}.sha256"

    sha256sum "$file" > "$checksum_file"
    echo "Checksum stored in $checksum_file"
}

# Verify checksum
verify_checksum() {
    local file="$1"
    local checksum_file="${file}.sha256"

    if [[ ! -f "$checksum_file" ]]; then
        echo "ERROR: Checksum file not found: $checksum_file"
        return 1
    fi

    sha256sum -c "$checksum_file"
}

# Encrypt and compress
encrypt_compress() {
    local input_file="$1"
    local output_file="${2:-${input_file}.gz.enc}"

    # Compress first
    gzip -c "$input_file" | \
    # Then encrypt
    openssl enc "-${ENCRYPTION_ALGORITHM}" \
        -salt \
        -out "$output_file" \
        -pass "file:${KEY_FILE}"

    # Calculate checksum of encrypted file
    calculate_checksum "$output_file"

    echo "File compressed and encrypted: $output_file"
}

# Decrypt and decompress
decrypt_decompress() {
    local input_file="$1"
    local output_file="${2:-${input_file%.gz.enc}}"

    # Verify checksum first
    if ! verify_checksum "$input_file"; then
        echo "ERROR: Checksum verification failed"
        return 1
    fi

    # Decrypt and decompress
    openssl enc "-${ENCRYPTION_ALGORITHM}" \
        -d \
        -in "$input_file" \
        -pass "file:${KEY_FILE}" | \
    gunzip -c > "$output_file"

    echo "File decrypted and decompressed: $output_file"
}

# Secure delete
secure_delete() {
    local file="$1"

    if command -v shred &> /dev/null; then
        shred -vfz -n 3 "$file"
    else
        # Fallback to dd if shred not available
        dd if=/dev/urandom of="$file" bs=1024 count=$(du -k "$file" | cut -f1) 2>/dev/null
        rm -f "$file"
    fi

    echo "File securely deleted: $file"
}

# Export functions for use in other scripts
export -f generate_key
export -f encrypt_file
export -f decrypt_file
export -f gpg_encrypt
export -f gpg_decrypt
export -f calculate_checksum
export -f verify_checksum
export -f encrypt_compress
export -f decrypt_decompress
export -f secure_delete
