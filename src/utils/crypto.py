"""Fernet-based encryption for tenant credentials.

All sensitive credentials (API keys, tokens) are encrypted at rest
using a symmetric key from the TENANT_ENCRYPTION_KEY env var.
"""

import structlog
from cryptography.fernet import Fernet

from config.settings import settings

log = structlog.get_logger()


def _get_fernet() -> Fernet:
    """Return a Fernet instance using the configured encryption key."""
    key = settings.tenant_encryption_key
    if not key:
        raise ValueError(
            "TENANT_ENCRYPTION_KEY not set. Generate one with: "
            "python -c 'from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())'"
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_value(plaintext: str) -> str:
    """Encrypt a plaintext string and return base64-encoded ciphertext.

    Args:
        plaintext: The sensitive value to encrypt.

    Returns:
        Fernet-encrypted base64 string.
    """
    f = _get_fernet()
    return f.encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a Fernet-encrypted base64 string.

    Args:
        ciphertext: The encrypted value from the database.

    Returns:
        Original plaintext string.

    Raises:
        InvalidToken: If the ciphertext is corrupted or the key is wrong.
    """
    f = _get_fernet()
    return f.decrypt(ciphertext.encode()).decode()


def mask_credential(value: str) -> str:
    """Mask a credential for display: show first 4 + last 4 chars.

    Args:
        value: Decrypted credential string.

    Returns:
        Masked string like "APCA...x4f2".
    """
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"
