import asyncio
import json
from typing import Optional, Dict, Any, Annotated

import requests
from pydantic import Field

from agent_framework import ChatAgent, ai_function
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential


# ========= Cloud Symptom Checker Instructions =========

SYMPTOM_CHECKER_INSTRUCTIONS = """
You are a careful symptom-checker assistant for non-emergency triage.

General behavior:
- You are NOT a clinician. Do NOT provide medical diagnosis or prescribe treatment.
- First, check for red-flag symptoms (e.g., chest pain, trouble breathing, severe bleeding, stroke signs,
  one-sided weakness, confusion, fainting). If any are present, advise urgent/emergency care and STOP.
- If no red-flags, summarize key factors (age group, duration, severity), then provide:
  1) sensible next steps a layperson could take,
  2) clear guidance on when to contact a clinician,
  3) simple self-care advice if appropriate.
- Use plain language, under 8 bullets total.
- Always end with: "This is not medical advice."

Tool usage:
- When the user provides raw lab report text, or mentions “labs below” or “see labs”, 
  you MUST call the `summarize_lab_report` tool to convert the labs into structured data
  before giving your triage guidance.
- Use the tool result as context, but do NOT expose the raw JSON directly. 
  Instead, summarize the key abnormal findings in plain language.
""".strip()


# ========= Local Lab Summarizer (Foundry Local + Phi-4-mini) =========

FOUNDRY_LOCAL_BASE = "http://127.0.0.1:52403"      # from `foundry service status`
FOUNDRY_LOCAL_CHAT_URL = FOUNDRY_LOCAL_BASE + "/v1/chat/completions"

# This is the model id you confirmed works:
FOUNDRY_LOCAL_MODEL_ID = "Phi-4-mini-instruct-cuda-gpu:5"


LOCAL_LAB_SYSTEM_PROMPT = """
You are a medical lab report summarizer running locally on the user's machine.

You MUST respond with ONLY one valid JSON object. Do not include any explanation,
backticks, markdown, or text outside the JSON. The JSON must have this shape:

{
  "overall_assessment": "<short plain English summary>",
  "notable_abnormal_results": [
    {
      "test": "string",
      "value": "string",
      "unit": "string or null",
      "reference_range": "string or null",
      "severity": "mild|moderate|severe"
    }
  ]
}

If you are unsure about a field, use null. Do NOT invent values.
""".strip()


def _strip_code_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` fences if present.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # remove leading ```
        stripped = stripped[3:].lstrip()
        # optional language tag like "json"
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
        # remove trailing ```
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


@ai_function(
    name="summarize_lab_report",
    description=(
        "Summarize a raw lab report into structured abnormalities using a local model "
        "running on the user's GPU. Use this whenever the user provides lab results as text."
    ),
)
def summarize_lab_report(
    lab_text: Annotated[str, Field(description="The raw text of the lab report to summarize.")],
) -> Dict[str, Any]:
    """
    Tool: summarize a lab report using Foundry Local (Phi-4-mini) on the user's GPU.

    Returns a JSON-compatible dict with:
    - overall_assessment: short text summary
    - notable_abnormal_results: list of abnormal test objects
    """

    payload = {
        "model": FOUNDRY_LOCAL_MODEL_ID,
        "messages": [
            {"role": "system", "content": LOCAL_LAB_SYSTEM_PROMPT},
            {"role": "user", "content": lab_text},
        ],
        "max_tokens": 256,
        "temperature": 0.2,
    }

    headers = {
        "Content-Type": "application/json",
    }

    print(f"[LOCAL TOOL] POST {FOUNDRY_LOCAL_CHAT_URL}")
    resp = requests.post(
        FOUNDRY_LOCAL_CHAT_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=120,
    )

    resp.raise_for_status()
    data = resp.json()

    # OpenAI-compatible shape: choices[0].message.content
    content = data["choices"][0]["message"]["content"]

    # Handle string vs list-of-parts
    if isinstance(content, list):
        content_text = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    else:
        content_text = content

    print("[LOCAL TOOL] Raw content from model:")
    print(content_text)

    # Strip ```json fences if present, then parse JSON
    cleaned = _strip_code_fences(content_text)
    lab_summary = json.loads(cleaned)
    print("[LOCAL TOOL] Parsed lab summary JSON:")
    print(json.dumps(lab_summary, indent=2))

    # Return dict – Agent Framework will serialize this as the tool result
    return lab_summary


# ========= Hybrid Main (Agent uses the local tool) =========

async def main():
    # Example free-text case + raw lab text that the agent can decide to send to the tool
    case = (
        "Teenager with bad headache and throwing up. Fever of 40C and no other symptoms."
    )

    lab_report_text = """
   -------------------------------------------
   SAN DIEGO FAMILY LABORATORY SERVICES
        4420 Camino Del Rio S, Suite 210
             San Diego, CA 92108
         Phone: (858) 555-4821  |  Fax: (858) 555-4822
    -------------------------------------------

    PATIENT INFORMATION
    Name:       Alex Thompson
    DOB:        04/12/2007 (17 yrs)
    Sex:        Male
    Patient ID: AXT-442871
    Address:    1921 Hawthorne Ridge Ct, Encinitas, CA 92024

    ORDERING PROVIDER
    Dr. Melissa Ortega, MD
    NPI: 1780952216
    Clinic: North County Pediatrics Group

    REPORT DETAILS
    Accession #: 24-SDFLS-118392
    Collected:   11/14/2025 14:32
    Received:    11/14/2025 16:06
    Reported:    11/14/2025 20:54
    Specimen:    Whole Blood (EDTA), Serum Separator Tube

    ------------------------------------------------------
    COMPLETE BLOOD COUNT (CBC)
    ------------------------------------------------------
    WBC ................. 14.5     x10^3/µL      (4.0 – 10.0)     HIGH
    RBC ................. 4.61     x10^6/µL      (4.50 – 5.90)
    Hemoglobin .......... 13.2     g/dL          (13.0 – 17.5)    LOW-NORMAL
    Hematocrit .......... 39.8     %             (40.0 – 52.0)    LOW
    MCV ................. 86.4     fL            (80 – 100)
    Platelets ........... 210      x10^3/µL      (150 – 400)

    ------------------------------------------------------
    INFLAMMATORY MARKERS
    ------------------------------------------------------
    C-Reactive Protein (CRP) ......... 60 mg/L       (< 5 mg/L)     HIGH
    Erythrocyte Sedimentation Rate ... 32 mm/hr      (0 – 15 mm/hr) HIGH

    ------------------------------------------------------
    BASIC METABOLIC PANEL (BMP)
    ------------------------------------------------------
    Sodium (Na) .............. 138   mmol/L       (135 – 145)
    Potassium (K) ............ 3.9   mmol/L       (3.5 – 5.1)
    Chloride (Cl) ............ 102   mmol/L       (98 – 107)
    CO2 (Bicarbonate) ........ 23    mmol/L       (22 – 29)
    Blood Urea Nitrogen (BUN)  11    mg/dL        (7 – 20)
    Creatinine ................ 0.74 mg/dL        (0.50 – 1.00)
    Glucose (fasting) ......... 109  mg/dL        (70 – 99)        HIGH

    ------------------------------------------------------
    LIVER FUNCTION TESTS
    ------------------------------------------------------
    AST ....................... 28  U/L          (0 – 40)
    ALT ....................... 22  U/L          (0 – 44)
    Alkaline Phosphatase ...... 144 U/L          (65 – 260)
    Total Bilirubin ........... 0.6 mg/dL        (0.1 – 1.2)

    ------------------------------------------------------
    NOTES
    ------------------------------------------------------
    Mild leukocytosis and elevated inflammatory markers (CRP, ESR) may indicate an acute
    infectious or inflammatory process. Glucose slightly elevated; could be non-fasting.

    ------------------------------------------------------
    END OF REPORT
    SDFLS-CLIA ID: 05D5554973
    This report is for informational purposes only and not a diagnosis.
------------------------------------------------------

    """

    # Single user message that gives both the case and labs.
    # The agent will see that there are labs and call summarize_lab_report() as a tool.
    user_message = (
        "Patient case:\n"
        f"{case}\n\n"
        "Here are the lab results as raw text. If helpful, you can summarize them first:\n"
        f"{lab_report_text}\n\n"
        "Please provide non-emergency triage guidance."
    )

    async with (
        AzureCliCredential() as credential,
        ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=credential),
            instructions=SYMPTOM_CHECKER_INSTRUCTIONS,
            # 👇 Tool is now attached to the agent
            tools=[summarize_lab_report],
            name="hybrid-symptom-checker",
        ) as agent,
    ):
        result = await agent.run(user_message)

        print("\n=== Symptom Checker (Hybrid: Local Tool + Cloud Agent) ===\n")
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
