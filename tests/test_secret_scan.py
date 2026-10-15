import subprocess

import pytest

from codepilot.secret_scan import scan_change

# Built at runtime so this file holds no literal token for scanners (or GitHub push protection) to flag.
ANTHROPIC = "sk-ant-" + "api03-" + "x7" * 20
HELD = "held" + "-value-" + "9f3k2m8q"


@pytest.fixture
def sandbox(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("A = 1\nB = 2\nOLD = 'gone soon " + ANTHROPIC + "'\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return repo


def test_clean_change_and_removed_secrets_pass(sandbox):
    (sandbox / "app.py").write_text("A = 1\nB = 3\n")  # removes the line holding a token
    assert scan_change(sandbox, {"PR body": "Fixes B."}, environ={"API_TOKEN": "short", "HOME": HELD}) == []


def test_held_secret_values_are_found_by_file_line_without_echoing_them(sandbox):
    (sandbox / "app.py").write_text(f"A = 1\nB = 2\nOLD = 'x'\n\nKEY = '{HELD}'\n")
    findings = scan_change(sandbox, {}, environ={"GROQ_API_KEY": HELD})
    assert [str(f) for f in findings] == ["app.py:5: value of $GROQ_API_KEY"]
    assert HELD not in str(findings)


def test_token_formats_in_new_files_and_pr_text_are_found(sandbox):
    (sandbox / "config").mkdir()
    (sandbox / "config" / "deploy.pem").write_text("-----BEGIN RSA " + "PRIVATE KEY-----\nabc\n")
    findings = scan_change(sandbox, {"PR body": f"Summary\nused {ANTHROPIC} to test"}, environ={})
    assert {str(f) for f in findings} == {"config/deploy.pem:1: private key", "PR body:2: Anthropic API key"}


def test_added_lines_that_look_like_diff_headers_are_still_scanned(sandbox):
    (sandbox / "app.py").write_text(f"A = 1\nB = 2\nOLD = 'x'\n++ {HELD}\n")
    assert [str(f) for f in scan_change(sandbox, {}, environ={"SECRET": HELD})] == ["app.py:4: value of $SECRET"]


def test_multi_line_secrets_are_matched_line_by_line(sandbox):
    body = "MIIEpAIBAAKCAQEA" + "q1w2e3r4" * 6
    (sandbox / "app.py").write_text(f"A = 1\nB = 2\nOLD = 'x'\nK = '{body}'\n")
    pem = "-----BEGIN RSA " + f"PRIVATE KEY-----\n{body}\n-----END RSA " + "PRIVATE KEY-----"
    assert [str(f) for f in scan_change(sandbox, {}, environ={"APP_PRIVATE_KEY": pem})] == [
        "app.py:4: value of $APP_PRIVATE_KEY"]
