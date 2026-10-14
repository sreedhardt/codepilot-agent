"""Hidden acceptance test for bookshelf-api#3 (loan renewal). Never shown to the agent."""

from datetime import date, timedelta

from app.config import LOAN_PERIOD_DAYS
from tests.conftest import TODAY


def checkout(client):
    return client.post("/loans", json={"book_id": 1, "member_id": 1}).json()


def test_renew_extends_from_current_due_date(client, today):
    loan = checkout(client)
    today.value = TODAY + timedelta(days=3)  # renew early: new due date must not be based on today
    response = client.post(f"/loans/{loan['id']}/renew")
    assert response.status_code == 200
    assert response.json()["due"] == (date.fromisoformat(loan["due"]) + timedelta(days=LOAN_PERIOD_DAYS)).isoformat()


def test_at_most_two_renewals(client):
    loan = checkout(client)
    assert client.post(f"/loans/{loan['id']}/renew").status_code == 200
    assert client.post(f"/loans/{loan['id']}/renew").status_code == 200
    assert client.post(f"/loans/{loan['id']}/renew").status_code == 409


def test_overdue_loan_cannot_be_renewed(client, today):
    loan = checkout(client)
    today.value = date.fromisoformat(loan["due"]) + timedelta(days=2)
    assert client.post(f"/loans/{loan['id']}/renew").status_code == 409


def test_returned_loan_cannot_be_renewed(client):
    loan = checkout(client)
    client.post(f"/loans/{loan['id']}/return")
    assert client.post(f"/loans/{loan['id']}/renew").status_code == 409


def test_unknown_loan_is_404(client):
    assert client.post("/loans/999/renew").status_code == 404
