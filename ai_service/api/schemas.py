"""
SalahSetu AI Service - Pydantic Request & Response Schemas
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class LegalQueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="Natural language legal query regarding Bharatiya Nyaya Sanhita (BNS)",
        example="What is cheating under the Bharatiya Nyaya Sanhita?",
        max_length=1000
    )

    @field_validator("question")

    @classmethod
    def validate_question_not_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Question cannot be empty or whitespace-only.")
        return v.strip()


class LegalSourceMetadata(BaseModel):
    section_number: str
    section_title: str
    source_page: Optional[int] = None
    source_authority: str = "India Code"
    source_url: Optional[str] = ""


class LegalQueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[LegalSourceMetadata]
    limitations: str
    retrieved_sections: List[str]


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadinessResponse(BaseModel):
    status: str
    artifacts_ready: bool
    details: dict
