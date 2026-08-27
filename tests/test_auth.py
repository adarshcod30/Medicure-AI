"""
Tests for authentication and history.

Everything here runs without MongoDB. The paths that matter most when storage
is absent — honest 503s rather than 500s, and 401s that leak nothing — are
exactly the paths a database-backed test suite would skip, so they are tested
directly against a service with `state.store` unset.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.auth import create_token, hash_password, verify_password
from apps.api.auth.security import decode_token
from apps.api.config import get_settings

SECRET = "x" * 40


@pytest.fixture
def client(monkeypatch):
    """A client whose service has no storage configured."""
    from apps.api import deps, main

    monkeypatch.setattr(deps.state, "store", None, raising=False)
    # Skip the lifespan: it loads a 125 MB index and pings Mongo, neither of
    # which any test in this file exercises.
    return TestClient(main.app)


# --- password hashing -----------------------------------------------------


def test_hash_is_salted_so_equal_passwords_differ():
    """Two users with the same password must not share a hash."""
    assert hash_password("correct horse") != hash_password("correct horse")


def test_verify_accepts_the_right_password_and_rejects_others():
    stored = hash_password("correct horse")
    assert verify_password("correct horse", stored)
    assert not verify_password("Correct Horse", stored)
    assert not verify_password("", stored)


def test_overlong_password_is_refused_rather_than_truncated():
    """bcrypt silently ignores bytes past 72.

    Truncating would make "<72 identical chars>A" and "<the same>B" the same
    credential, so the limit is enforced instead of hidden.
    """
    with pytest.raises(ValueError, match="72 bytes"):
        hash_password("a" * 73)


def test_malformed_stored_hash_is_a_mismatch_not_a_crash():
    assert verify_password("anything", "not-a-bcrypt-hash") is False


# --- tokens ---------------------------------------------------------------


def test_token_round_trips_subject_and_email():
    token = create_token(
        subject="abc123", email="a@b.co", secret=SECRET, algorithm="HS256", expiry_minutes=60
    )
    claims = decode_token(token, secret=SECRET, algorithm="HS256")
    assert claims["sub"] == "abc123"
    assert claims["email"] == "a@b.co"


def test_token_signed_with_another_secret_is_rejected():
    token = create_token(
        subject="abc123", email="a@b.co", secret=SECRET, algorithm="HS256", expiry_minutes=60
    )
    assert decode_token(token, secret="y" * 40, algorithm="HS256") is None


def test_expired_token_is_rejected():
    token = create_token(
        subject="abc123", email="a@b.co", secret=SECRET, algorithm="HS256", expiry_minutes=-1
    )
    assert decode_token(token, secret=SECRET, algorithm="HS256") is None


def test_garbage_token_is_rejected_without_raising():
    assert decode_token("not.a.token", secret=SECRET, algorithm="HS256") is None


# --- routes without a database --------------------------------------------


def _bearer() -> dict:
    settings = get_settings()
    token = create_token(
        subject="507f1f77bcf86cd799439011",
        email="a@b.co",
        secret=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
        expiry_minutes=60,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/v1/auth/register", {"email": "a@b.co", "password": "abcd1234"}),
        ("post", "/v1/auth/login", {"email": "a@b.co", "password": "abcd1234"}),
    ],
)
def test_auth_routes_report_503_when_storage_is_absent(client, method, path, body):
    """503 and a plain explanation — not 500, and not a silent failure.

    Running without MongoDB is a supported configuration, so the status code
    has to distinguish "this deployment does not offer accounts" from "this
    service is broken".
    """
    response = getattr(client, method)(path, json=body)
    assert response.status_code == 503
    assert "no database is configured" in response.json()["detail"]


def test_history_requires_authentication_before_it_checks_storage(client):
    """401 for an anonymous caller, even with storage off.

    Order matters: reporting 503 first would tell an unauthenticated stranger
    about the deployment's configuration.
    """
    assert client.get("/v1/history").status_code == 401


def test_history_reports_503_for_an_authenticated_caller(client):
    response = client.get("/v1/history", headers=_bearer())
    assert response.status_code == 503


def test_me_requires_authentication(client):
    assert client.get("/v1/auth/me").status_code == 401


def test_bad_token_is_401_not_500(client):
    response = client.get("/v1/history", headers={"Authorization": "Bearer garbage"})
    assert response.status_code == 401


def test_short_password_is_rejected_by_validation(client):
    """Refused at the schema, before any storage check."""
    response = client.post(
        "/v1/auth/register", json={"email": "a@b.co", "password": "short"}
    )
    assert response.status_code == 422


def test_malformed_email_is_rejected_by_validation(client):
    response = client.post(
        "/v1/auth/register", json={"email": "not-an-email", "password": "abcd1234"}
    )
    assert response.status_code == 422
