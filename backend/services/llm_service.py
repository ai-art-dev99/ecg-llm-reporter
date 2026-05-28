"""
LLM Service — Groq API
-----------------------
Generates structured ECG clinical reports using:
  - ECG features (from ecg_processor.py)
  - RAG context (from rag/retriever.py)
  - Groq API (llama-3.3-70b-versatile)

Prompt strategy:
  System  → cardiologist persona + output format
  User    → ECG measurements + retrieved guidelines + task
"""

import json
import os
import re
import time
from typing import Optional

from groq import Groq

from services.report_schema import ECGReport, ReportSection, Urgency


# ─── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert cardiologist AI assistant specialising in ECG interpretation.
Your role is to generate structured, clinically accurate ECG analysis reports.

IMPORTANT RULES:
1. Base your analysis ONLY on the provided ECG measurements and guidelines.
2. Always note that this is an AI-assisted analysis and requires clinical correlation.
3. Be specific about measurements — reference actual numbers from the data.
4. Flag any urgent findings immediately and clearly.
5. Use standard cardiology terminology.
6. You MUST respond with valid JSON only — no markdown, no explanation outside JSON.

OUTPUT FORMAT (strict JSON):
{
  "primary_diagnosis": "string — main diagnosis in one sentence",
  "urgency": "routine" | "urgent" | "emergency",
  "confidence": "high" | "moderate" | "low",
  "key_abnormalities": ["list", "of", "findings"],
  "report": {
    "findings": "Objective description of ECG measurements",
    "interpretation": "Clinical meaning of the findings",
    "differentials": "Other diagnoses to consider",
    "recommendations": "Clinical next steps",
    "limitations": "Caveats and limitations of this AI analysis"
  },
  "rag_sources_used": ["list of guideline sources referenced"]
}"""


def build_user_prompt(ecg_context: str, rag_context: str) -> str:
    return f"""Analyse the following ECG and generate a structured clinical report.

{ecg_context}

---

{rag_context}

---

Based on the ECG measurements above and the retrieved clinical guidelines,
generate a complete structured ECG report in the specified JSON format.
Reference specific measurements (heart rate, intervals, etc.) in your findings.
"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_str(val) -> str:
    """
    Normalise LLM output field to string.
    LLMs sometimes return lists instead of strings — join them.
    """
    if isinstance(val, list):
        return " ".join(str(item) for item in val)
    return str(val) if val is not None else ""


# ─── LLM Service ─────────────────────────────────────────────────────────────

class LLMService:
    """
    Groq-powered ECG report generator.

    Usage:
        service = LLMService()
        report = service.generate_report(features, retrieval_result)
    """

    MODEL = "llama-3.3-70b-versatile"
    FALLBACK_MODEL = "llama-3.1-8b-instant"   # faster, lower quality
    MAX_TOKENS = 1500
    TEMPERATURE = 0.2   # low = more consistent clinical output

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY not set. "
                "Get a free key at https://console.groq.com — takes 30 seconds."
            )
        self.client = Groq(api_key=key)
        self._verify_connection()

    def _verify_connection(self):
        """Quick check that the API key is valid."""
        try:
            # Minimal test call
            self.client.chat.completions.create(
                model=self.FALLBACK_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            print(f"[LLM] Groq API connected. Model: {self.MODEL}")
        except Exception as e:
            raise ConnectionError(f"[LLM] Groq API connection failed: {e}")

    def generate_report(self, features, retrieval_result) -> ECGReport:
        """
        Main entry point.
        Takes ECGFeatures + RetrievalResult → returns ECGReport.
        """
        t0 = time.perf_counter()

        ecg_context = features.to_prompt_context()
        rag_context = retrieval_result.format_for_prompt(max_chars=2500)

        user_prompt = build_user_prompt(ecg_context, rag_context)

        raw_json = self._call_groq(user_prompt)
        parsed = self._parse_response(raw_json, features)
        parsed.processing_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        return parsed

    def generate_report_from_text(self, ecg_summary: str, guidelines: str = "") -> ECGReport:
        """
        Convenience method for testing — takes plain text instead of objects.
        """
        t0 = time.perf_counter()
        rag_ctx = f"RETRIEVED GUIDELINES\n{'='*40}\n{guidelines}" if guidelines else ""
        user_prompt = build_user_prompt(ecg_summary, rag_ctx)
        raw_json = self._call_groq(user_prompt)
        parsed = self._parse_response(raw_json, record_name="text_input")
        parsed.processing_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        return parsed

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_groq(self, user_prompt: str, use_fallback: bool = False) -> str:
        """Call Groq API and return raw response string."""
        model = self.FALLBACK_MODEL if use_fallback else self.MODEL
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                response_format={"type": "json_object"},  # force JSON output
            )
            return response.choices[0].message.content

        except Exception as e:
            if not use_fallback:
                print(f"[LLM] {model} failed ({e}), retrying with fallback model...")
                return self._call_groq(user_prompt, use_fallback=True)
            raise RuntimeError(f"[LLM] Both models failed: {e}")

    def _parse_response(self, raw: str, features=None, record_name: str = "unknown") -> ECGReport:
        """Parse JSON response into ECGReport. Falls back gracefully on errors."""
        # Strip any accidental markdown fences
        clean = re.sub(r"```json|```", "", raw).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            print(f"[LLM] JSON parse error: {e}\nRaw: {raw[:200]}")
            return self._fallback_report(features, record_name, raw)

        # Extract record name from features if available
        rec_name = getattr(features, "record_name", record_name)

        # Build ReportSection
        # LLM sometimes returns lists instead of strings — normalise all fields
        report_data = data.get("report", {})
        report_section = ReportSection(
            findings=_to_str(report_data.get("findings", "Not provided")),
            interpretation=_to_str(report_data.get("interpretation", "Not provided")),
            differentials=_to_str(report_data.get("differentials", "Not provided")),
            recommendations=_to_str(report_data.get("recommendations", "Not provided")),
            limitations=_to_str(report_data.get("limitations",
                "AI-generated report. Not a substitute for clinical assessment.")),
        )

        # Normalise urgency
        urgency_raw = data.get("urgency", "routine").lower()
        try:
            urgency = Urgency(urgency_raw)
        except ValueError:
            urgency = Urgency.ROUTINE

        return ECGReport(
            record_name=rec_name,
            model_used=self.MODEL,
            report=report_section,
            primary_diagnosis=data.get("primary_diagnosis", "Unable to determine"),
            urgency=urgency,
            confidence=data.get("confidence", "low"),
            key_abnormalities=data.get("key_abnormalities", []),
            rag_sources_used=data.get("rag_sources_used", []),
            processing_time_ms=0.0,
        )

    def _fallback_report(self, features, record_name: str, raw: str) -> ECGReport:
        """Return a minimal report when parsing fails."""
        rec_name = getattr(features, "record_name", record_name)
        return ECGReport(
            record_name=rec_name,
            model_used=self.MODEL,
            report=ReportSection(
                findings="LLM response could not be parsed.",
                interpretation="See raw response in limitations field.",
                differentials="N/A",
                recommendations="Please retry or review manually.",
                limitations=f"Parse error. Raw LLM output: {raw[:300]}",
            ),
            primary_diagnosis="Parse error — see limitations",
            urgency=Urgency.ROUTINE,
            confidence="low",
            key_abnormalities=[],
            rag_sources_used=[],
            processing_time_ms=0.0,
        )