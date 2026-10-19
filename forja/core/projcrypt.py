"""Projetos com palavra-passe: cifra AES (Fernet) com chave derivada da palavra-passe
(PBKDF2-SHA256, 390 000 iterações, sal aleatório)."""
import base64
import os

MAGIC = b"FORJAENC1"


def _key(password: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390_000)
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))


def encrypt(data: bytes, password: str) -> bytes:
    from cryptography.fernet import Fernet
    salt = os.urandom(16)
    return MAGIC + salt + Fernet(_key(password, salt)).encrypt(data)


def decrypt(blob: bytes, password: str) -> bytes:
    from cryptography.fernet import Fernet
    if not blob.startswith(MAGIC):
        raise ValueError("não é um projeto cifrado")
    salt = blob[len(MAGIC):len(MAGIC) + 16]
    return Fernet(_key(password, salt)).decrypt(blob[len(MAGIC) + 16:])


def is_encrypted(path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(MAGIC)) == MAGIC
    except Exception:
        return False
