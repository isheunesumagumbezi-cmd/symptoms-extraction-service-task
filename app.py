import os
from enum import Enum
from typing import Literal

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, ConfigDict


class SignId(str, Enum):
    REDUCED_APPETITE = "reduced_appetite"
    WEAKNESS = "weakness"
    SWELLING = "swelling"
    RAPID_BREATHING = "rapid_breathing"
    CLOUDY_EYE = "cloudy_eye"
    FROTH_NOSE_MOUTH = "froth_nose_mouth"
    RED_URINE = "red_urine"
    UNABLE_TO_STAND = "unable_to_stand"
    TICKS = "ticks"


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sign_id: SignId
    value: Literal["yes", "no", "not_sure"]
    quote: str


class LLMFindings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[Finding]


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str


class ExtractResponse(BaseModel):
    findings: list[Finding]
    escalate: bool
    reason: str


ESCALATION_SIGNS = {
    SignId.FROTH_NOSE_MOUTH,
    SignId.UNABLE_TO_STAND,
    SignId.CLOUDY_EYE,
}

SYSTEM_PROMPT = """Extract only the veterinary signs explicitly mentioned in the farmer's message.
Return a JSON object with exactly one key, "findings", containing a list of objects
with "sign_id", "value", and "quote". Use only these sign IDs:
- reduced_appetite: eating less, stopped eating, or not grazing
- weakness: dull, weak, or lying down more than usual
- swelling: lump below the ear, under the jaw, or in front of the shoulder; may be called mbandabvu
- rapid_breathing: breathing fast while resting
- cloudy_eye: eye looks white, grey, or milky
- froth_nose_mouth: foam or froth from the nose or mouth
- red_urine: urine is red, brown, or the colour of strong tea
- unable_to_stand: cannot get up or stay standing
- ticks: ticks can be seen on the animal

Use "yes" for a stated present sign, "no" for an explicitly denied sign, and
"not_sure" when the farmer expresses uncertainty about a mentioned sign. Do not
infer signs from context or report signs that are not mentioned. "Fine" or
"healthy" alone does not mean every sign was explicitly denied. Each quote must
be a verbatim, non-empty excerpt from the message that supports the finding.
Return an empty findings list if no listed signs are mentioned."""


def extract_findings(message: str) -> list[Finding]:
    """Ask the configured LLM for findings and validate them before use."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Extraction requires OPENAI_API_KEY. Set it in the server terminal and restart.",
        )
    client = OpenAI()
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    content = completion.choices[0].message.content
    if content is None:
        raise ValueError("The LLM returned an empty response.")

    parsed = LLMFindings.model_validate_json(content)
    seen_signs: set[SignId] = set()
    for finding in parsed.findings:
        if finding.sign_id in seen_signs:
            raise ValueError(f"The LLM returned duplicate sign_id: {finding.sign_id.value}")
        seen_signs.add(finding.sign_id)
        if not finding.quote or finding.quote not in message:
            raise ValueError(
                f"The quote for {finding.sign_id.value} is not a verbatim excerpt "
                "from the message."
            )
    return parsed.findings


def escalation_for(findings: list[Finding]) -> tuple[bool, str]:
    triggers = [
        finding
        for finding in findings
        if finding.sign_id in ESCALATION_SIGNS
        and finding.value in {"yes", "not_sure"}
    ]
    if not triggers:
        return False, "No escalation signs were reported as present or uncertain."

    details = ", ".join(
        f"{finding.sign_id.value} ({finding.value})" for finding in triggers
    )
    return True, f"Escalation signs reported as present or uncertain: {details}."


app = FastAPI(title="V-Vet Symptom Extraction API")


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    findings = extract_findings(request.message)
    escalate, reason = escalation_for(findings)
    return ExtractResponse(findings=findings, escalate=escalate, reason=reason)
