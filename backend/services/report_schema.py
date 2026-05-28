"""
ECG Report Schema
-----------------
Pydantic models for structured LLM output.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Urgency(str, Enum):
    ROUTINE   = "routine"
    URGENT    = "urgent"
    EMERGENCY = "emergency"


class ReportSection(BaseModel):
    findings:        str = Field(description="Objective ECG findings")
    interpretation:  str = Field(description="Clinical interpretation")
    differentials:   str = Field(description="Differential diagnoses to consider")
    recommendations: str = Field(description="Clinical recommendations")
    limitations:     str = Field(description="Limitations and caveats of this analysis")


class ECGReport(BaseModel):
    # Identity
    record_name:      str
    generated_at:     str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    model_used:       str

    # Structured sections
    report:           ReportSection

    # Summary fields
    primary_diagnosis:   str
    urgency:             Urgency
    confidence:          str          # high | moderate | low
    key_abnormalities:   list[str]

    # Context
    rag_sources_used:    list[str]    # which guidelines were referenced
    processing_time_ms:  float

    def to_text(self) -> str:
        """Format as readable clinical report."""
        lines = [
            "=" * 60,
            "       ECG ANALYSIS REPORT",
            "=" * 60,
            f"Record      : {self.record_name}",
            f"Generated   : {self.generated_at}",
            f"Model       : {self.model_used}",
            f"Urgency     : {self.urgency.value.upper()}",
            f"Confidence  : {self.confidence}",
            "",
            "PRIMARY DIAGNOSIS",
            "-" * 40,
            self.primary_diagnosis,
            "",
        ]

        if self.key_abnormalities:
            lines += ["KEY ABNORMALITIES", "-" * 40]
            for a in self.key_abnormalities:
                lines.append(f"  • {a}")
            lines.append("")

        lines += [
            "FINDINGS",        "-" * 40, self.report.findings,        "",
            "INTERPRETATION",  "-" * 40, self.report.interpretation,  "",
            "DIFFERENTIALS",   "-" * 40, self.report.differentials,   "",
            "RECOMMENDATIONS", "-" * 40, self.report.recommendations, "",
            "LIMITATIONS",     "-" * 40, self.report.limitations,     "",
        ]

        if self.rag_sources_used:
            lines += ["GUIDELINES REFERENCED", "-" * 40]
            for s in self.rag_sources_used:
                lines.append(f"  • {s}")

        lines.append("=" * 60)
        return "\n".join(lines)