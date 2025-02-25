from pydantic import BaseModel, Field # type: ignore
from typing import Annotated, List
from typing_extensions import Literal


class Decision(BaseModel):
    decision: Literal["ocr", "od", "sc"] = Field(
        description="Decide which tools to use. ocr is for OCR, od is for object detection, and sc is for scene captioning.",
    )
    reasoning: str = Field(
        description="Provide reasoning for the decision.",
    )