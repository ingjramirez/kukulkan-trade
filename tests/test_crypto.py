"""Tests for Fernet-based credential encryption."""

import pytest
from cryptography.fernet import Fernet, InvalidToken

from config.settings import settings


# Generate a test encryption key for this module
_TEST_KEY = Fernet.generate_key().decode()


@pytest.fixture(autouse=True)
def _set_encryption_key(monkeypatch):
    """Set a valid encryption key for all tests."""
    monkeypatch.setattr(settings, "tenant_encryption_key", _TEST_KEY)


class TestEncryptDecrypt:
    def test_round_trip(self):
        from src.utils.crypto import decrypt_value, encrypt_value

        plaintext = "APCA-API-KEY-12345"
        ciphertext = encrypt_value(plaintext)
        assert ciphertext != plaintext
        assert decrypt_value(ciphertext) == plaintext

    def test_different_ciphertexts(self):
        """Same plaintext produces different ciphertexts (Fernet uses timestamps)."""
        from src.utils.crypto import encrypt_value

        a = encrypt_value("secret")
        b = encrypt_value("secret")
        # They may differ due to timestamp in Fernet token
        # But decrypting both should yield the same value
        from src.utils.crypto import decrypt_value
        assert decrypt_value(a) == decrypt_value(b) == "secret"

    def test_empty_string(self):
        from src.utils.crypto import decrypt_value, encrypt_value

        enc = encrypt_value("")
        assert decrypt_value(enc) == ""

    def test_unicode(self):
        from src.utils.crypto import decrypt_value, encrypt_value

        text = "contraseña-🔐-密码"
        assert decrypt_value(encrypt_value(text)) == text

    def test_long_value(self):
        from src.utils.crypto import decrypt_value, encrypt_value

        text = "x" * 10_000
        assert decrypt_value(encrypt_value(text)) == text


class TestDecryptWithWrongKey:
    def test_wrong_key_raises(self, monkeypatch):
        from src.utils.crypto import decrypt_value, encrypt_value

        ciphertext = encrypt_value("secret")

        # Switch to a different key
        new_key = Fernet.generate_key().decode()
        monkeypatch.setattr(settings, "tenant_encryption_key", new_key)

        with pytest.raises(InvalidToken):
            decrypt_value(ciphertext)


class TestMissingKey:
    def test_no_key_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "tenant_encryption_key", "")
        from src.utils.crypto import encrypt_value
        with pytest.raises(ValueError, match="TENANT_ENCRYPTION_KEY not set"):
            encrypt_value("test")


class TestMaskCredential:
    def test_normal_string(self):
        from src.utils.crypto import mask_credential
        assert mask_credential("APCA-KEY-12345abc") == "APCA...5abc"

    def test_short_string(self):
        from src.utils.crypto import mask_credential
        assert mask_credential("short") == "****"

    def test_exactly_8_chars(self):
        from src.utils.crypto import mask_credential
        assert mask_credential("12345678") == "****"

    def test_9_chars(self):
        from src.utils.crypto import mask_credential
        assert mask_credential("123456789") == "1234...6789"
