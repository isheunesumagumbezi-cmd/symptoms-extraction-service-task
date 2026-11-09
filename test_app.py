from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app as symptom_app


client = TestClient(symptom_app.app)


@pytest.fixture(autouse=True)
def set_test_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def finding(sign_id: str, value: str, quote: str) -> symptom_app.Finding:
    return symptom_app.Finding(sign_id=sign_id, value=value, quote=quote)


def test_extract_returns_findings_and_escalates_for_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    message = "There's foam coming from her nose and she can't get up."
    findings = [
        finding("froth_nose_mouth", "yes", "foam coming from her nose"),
        finding("unable_to_stand", "yes", "can't get up"),
    ]
    monkeypatch.setattr(symptom_app, "extract_findings", lambda _: findings)

    response = client.post("/extract", json={"message": message})

    assert response.status_code == 200
    assert response.json() == {
        "findings": [
            {"sign_id": "froth_nose_mouth", "value": "yes", "quote": "foam coming from her nose"},
            {"sign_id": "unable_to_stand", "value": "yes", "quote": "can't get up"},
        ],
        "escalate": True,
        "reason": (
            "Escalation signs reported as present or uncertain: "
            "froth_nose_mouth (yes), unable_to_stand (yes)."
        ),
    }


def test_uncertain_escalation_sign_triggers_but_negative_does_not() -> None:
    uncertain = finding("cloudy_eye", "not_sure", "might look cloudy")
    negative = finding("froth_nose_mouth", "no", "no foam")

    assert symptom_app.escalation_for([uncertain])[0] is True
    assert symptom_app.escalation_for([negative])[0] is False


def test_non_escalation_signs_do_not_escalate() -> None:
    findings = [
        finding("reduced_appetite", "yes", "not been eating"),
        finding("rapid_breathing", "yes", "breathing very fast"),
    ]

    assert symptom_app.escalation_for(findings)[0] is False


def test_llm_output_is_validated_and_only_returns_supported_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = (
        '{"findings":[{"sign_id":"ticks","value":"yes",'
        '"quote":"lots of ticks on the ears"}]}'
    )
    mock_client = Mock()
    mock_client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content=content))
    ]
    monkeypatch.setattr(symptom_app, "OpenAI", lambda: mock_client)
    message = "The calf is fine, just lots of ticks on the ears."

    result = symptom_app.extract_findings(message)

    assert result == [finding("ticks", "yes", "lots of ticks on the ears")]
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize(
    "content",
    [
        '{"findings":[{"sign_id":"made_up","value":"yes","quote":"sick"}]}',
        '{"findings":[{"sign_id":"ticks","value":"maybe","quote":"ticks"}]}',
        '{"findings":[{"sign_id":"ticks","value":"yes","quote":"not in message"}]}',
        (
            '{"findings":['
            '{"sign_id":"ticks","value":"yes","quote":"ticks"},'
            '{"sign_id":"ticks","value":"no","quote":"no ticks"}]}'
        ),
    ],
)
def test_invalid_or_unsupported_llm_output_is_rejected(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    mock_client = Mock()
    mock_client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content=content))
    ]
    monkeypatch.setattr(symptom_app, "OpenAI", lambda: mock_client)

    with pytest.raises((ValidationError, ValueError)):
        symptom_app.extract_findings("The animal has ticks.")


def test_extract_requires_message() -> None:
    response = client.post("/extract", json={})

    assert response.status_code == 422


def test_extract_reports_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")

    response = client.post("/extract", json={"message": "My cow is not eating."})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]
