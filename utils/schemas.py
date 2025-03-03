from pydantic import BaseModel, Field # type: ignore
from typing import Annotated, List
from typing_extensions import Literal

# Schema for structured output to use in planning

class AnswerFeedback(BaseModel):
    grade: Literal["accepted", "not accepted"] = Field(
        description="Decide if the answer is logical or not.",
    )
    feedback: str = Field(
        description="If the answer is not logical, provide feedback on how to improve it.",
    )

class Decision(BaseModel):
    decision: Literal["ocr", "od", "sc"] = Field(
        description="Decide which tools to use. ocr is for OCR, od is for object detection, and sc is for scene captioning.",
    )
    reasoning: str = Field(
        description="Provide reasoning for the decision.",
    )