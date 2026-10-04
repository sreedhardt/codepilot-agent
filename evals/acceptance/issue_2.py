"""Hidden acceptance test for bookshelf-api#2 (case-sensitive search). Never shown to the agent."""

import pytest


@pytest.mark.parametrize("query, expected", [("dune", 1), ("DUNE", 1), ("le guin", 2), ("LE GUIN", 2), ("Dune", 1)])
def test_search_ignores_case(client, query, expected):
    assert client.get("/books", params={"q": query}).json()["total"] == expected


def test_search_still_filters(client):
    assert client.get("/books", params={"q": "zzz-no-match"}).json()["total"] == 0
