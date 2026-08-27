"""
Tests for the interaction engine and the cabinet.

Every drug in these fixtures is invented — testium, mockazole, placebonol.
That is deliberate and load-bearing: this file must not become a place where
real clinical claims enter the repository by way of a test, because a fixture
is exactly the kind of "obviously fine" exception that erodes the rule the
engine exists to enforce. Real data arrives only through
scripts/ingest_interactions.py, from DDInter, with citations attached.
"""

from __future__ import annotations

import csv

import pytest
from fastapi.testclient import TestClient

from packages.pharmacology.interactions import (
    InteractionTable,
    check_signatures,
    ingredients_of,
)

SIG_TESTIUM = (("testium", 500.0, "mg", None),)
SIG_MOCKAZOLE = (("mockazole", 250.0, "mg", None),)
SIG_PLACEBONOL = (("placebonol", 10.0, "mg", None),)
SIG_COMBO = (("testium", 250.0, "mg", None), ("placebonol", 5.0, "mg", None))


@pytest.fixture
def table(tmp_path) -> InteractionTable:
    """A table over invented drugs."""
    directory = tmp_path / "interactions"
    directory.mkdir()
    with (directory / "interactions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ingredient_a", "ingredient_b", "severity", "description",
                "ddinter_id_a", "ddinter_id_b", "source_url",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "ingredient_a": "testium", "ingredient_b": "mockazole",
            "severity": "Major", "description": "Fictional example interaction.",
            "ddinter_id_a": "DDI-FAKE-1", "ddinter_id_b": "DDI-FAKE-2",
            "source_url": "https://example.invalid/",
        })
        writer.writerow({
            "ingredient_a": "placebonol", "ingredient_b": "mockazole",
            "severity": "Minor", "description": "Another fictional example.",
            "ddinter_id_a": "DDI-FAKE-3", "ddinter_id_b": "DDI-FAKE-2",
            "source_url": "https://example.invalid/",
        })
    return InteractionTable(tmp_path)


@pytest.fixture
def empty_table(tmp_path) -> InteractionTable:
    return InteractionTable(tmp_path)


# --- loading and availability --------------------------------------------


def test_a_missing_dataset_is_reported_not_raised(empty_table):
    """Absent data is a state, not an exception — the same contract as Bedrock."""
    assert empty_table.available is False
    assert "ingest_interactions" in empty_table.last_error
    assert empty_table.status()["pairs"] == 0


def test_a_present_dataset_reports_its_size(table):
    assert table.available is True
    assert len(table) == 2


# --- lookup ---------------------------------------------------------------


def test_lookup_is_order_independent(table):
    assert table.lookup("testium", "mockazole") is not None
    assert table.lookup("mockazole", "testium") is not None


def test_lookup_folds_salt_forms_to_the_canonical_name(table):
    """The dataset and the catalogue must meet at the same canonical form.

    If they do not, every lookup silently misses and the engine reports "no
    interactions" for a pair it holds data on — under-reporting, which is the
    dangerous direction.
    """
    assert table.lookup("testium hydrochloride", "mockazole") is not None


def test_unknown_pairs_return_none(table):
    assert table.lookup("testium", "placebonol") is None


# --- checking -------------------------------------------------------------


def test_a_known_pair_is_found_with_provenance(table):
    result = check_signatures([SIG_TESTIUM, SIG_MOCKAZOLE], table=table)
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["severity"] == "Major"
    assert finding["source"]["dataset"] == "ddinter"
    assert finding["source"]["record_id"]


def test_findings_are_sorted_most_severe_first(table):
    """A truncating UI must not be what hides the Major finding."""
    result = check_signatures(
        [SIG_TESTIUM, SIG_MOCKAZOLE, SIG_PLACEBONOL], table=table
    )
    severities = [f["severity"] for f in result["findings"]]
    assert severities == ["Major", "Minor"]


def test_no_findings_says_absence_of_evidence_not_safety(table):
    result = check_signatures([SIG_TESTIUM, SIG_PLACEBONOL], table=table)
    assert result["findings"] == []
    assert "absence of evidence" in result["coverage_note"]
    assert "safety" in result["coverage_note"]


def test_a_single_item_has_nothing_to_check(table):
    result = check_signatures([SIG_TESTIUM], table=table)
    assert result["findings"] == []
    assert result["ingredient_pairs_checked"] == 0


def test_an_empty_cabinet_is_handled(table):
    result = check_signatures([], table=table)
    assert result["findings"] == []
    assert result["duplicate_therapy"] == []


# --- duplicate therapy (needs no dataset) ---------------------------------


def test_duplicate_ingredient_is_detected_without_any_dataset(empty_table):
    """Arithmetic over the resolver's output, so it survives a missing dataset."""
    result = check_signatures(
        [SIG_TESTIUM, SIG_COMBO], labels=["Brand One", "Brand Two"], table=empty_table
    )
    duplicates = result["duplicate_therapy"]
    assert len(duplicates) == 1
    assert duplicates[0]["ingredient"] == "testium"
    assert set(duplicates[0]["items"]) == {"Brand One", "Brand Two"}


def test_distinct_ingredients_produce_no_duplicate_warning(empty_table):
    result = check_signatures([SIG_TESTIUM, SIG_MOCKAZOLE], table=empty_table)
    assert result["duplicate_therapy"] == []


def test_missing_dataset_says_so_and_still_reports_duplicates(empty_table):
    result = check_signatures([SIG_TESTIUM, SIG_COMBO], table=empty_table)
    assert result["available"] is False
    assert "No interaction dataset is installed" in result["coverage_note"]
    assert result["duplicate_therapy"]


def test_ingredients_are_folded_from_signatures():
    assert ingredients_of(SIG_COMBO) == ["testium", "placebonol"]


# --- routes ---------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    from apps.api import deps, main

    monkeypatch.setattr(deps.state, "store", None, raising=False)
    return TestClient(main.app)


def test_stateless_check_needs_no_account(client):
    """It stores nothing, so it requires nothing."""
    response = client.post(
        "/v1/interactions/check",
        json={"signatures": [[["testium", 500.0, "mg", None]]]},
    )
    assert response.status_code == 200
    assert "coverage_note" in response.json()


def test_mismatched_labels_are_rejected(client):
    response = client.post(
        "/v1/interactions/check",
        json={
            "signatures": [[["testium", 500.0, "mg", None]]],
            "labels": ["one", "two"],
        },
    )
    assert response.status_code == 422


def test_cabinet_requires_authentication(client):
    assert client.get("/v1/cabinet").status_code == 401
    assert client.post("/v1/cabinet", json={"display_name": "x", "signature": [["a"]]}).status_code == 401


def test_interactions_status_is_public(client):
    response = client.get("/v1/interactions/status")
    assert response.status_code == 200
    assert "available" in response.json()
