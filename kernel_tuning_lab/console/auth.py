"""Persistent workspace passwords; credentials are never generated or silently replaced."""
from __future__ import annotations

import hashlib
import hmac
import secrets

PASSWORD_KEY = 'login_password_hash'
ITERATIONS = 600_000


def verify_password(password, encoded):
    if not isinstance(password, str) or not password or len(password.encode('utf-8')) > 1024:
        return False
    try:
        algorithm, iterations, salt, expected = encoded.split('$')
        rounds = int(iterations)
        salt, expected = bytes.fromhex(salt), bytes.fromhex(expected)
        if algorithm != 'pbkdf2_sha256' or not 1 <= rounds <= 1_000_000 or len(salt) != 16 or len(expected) != 32:
            return False
        actual = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, rounds)
        return hmac.compare_digest(actual, expected)
    except (AttributeError, TypeError, ValueError):
        return False


def initialize_password(store, password):
    """Set once; supplying a different password later is an error, not a reset."""
    if not isinstance(password, str) or not password or len(password.encode('utf-8')) > 1024:
        raise ValueError('provide a nonempty workspace password of at most 1024 UTF-8 bytes')
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT value FROM metadata WHERE key=?', (PASSWORD_KEY,)).fetchone()
        if row:
            if not verify_password(password, row[0]):
                raise ValueError('workspace password is already configured; it was not changed')
            return False
        salt = secrets.token_bytes(16)
        result = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, ITERATIONS)
        encoded = f'pbkdf2_sha256${ITERATIONS}${salt.hex()}${result.hex()}'
        db.execute('INSERT INTO metadata VALUES (?,?)', (PASSWORD_KEY, encoded))
        # Switching from token login invalidates previous browser sessions once.
        db.execute('UPDATE metadata SET value=? WHERE key=?', (secrets.token_hex(32), 'session_secret'))
    return True
