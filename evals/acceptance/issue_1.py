"""Hidden acceptance test for bookshelf-api#1 (overdue on the due date). Never shown to the agent."""

from datetime import date, timedelta

from tests.conftest import TODAY


def test_can_borrow_on_due_date(client, today):
    loan = client.post("/loans", json={"book_id": 1, "member_id": 1}).json()
    today.value = date.fromisoformat(loan["due"])
    assert client.post("/loans", json={"book_id": 2, "member_id": 1}).status_code == 201


def test_not_overdue_on_due_date_but_overdue_day_after(client, today):
    loan = client.post("/loans", json={"book_id": 1, "member_id": 1}).json()
    due = date.fromisoformat(loan["due"])

    today.value = due
    assert client.get("/members/1/loans").json()[0]["overdue"] is False

    today.value = due + timedelta(days=1)
    assert client.get("/members/1/loans").json()[0]["overdue"] is True


def test_still_blocked_day_after_due_date(client, today):
    client.post("/loans", json={"book_id": 1, "member_id": 1})
    today.value = TODAY + timedelta(days=15)
    assert client.post("/loans", json={"book_id": 2, "member_id": 1}).status_code == 409
