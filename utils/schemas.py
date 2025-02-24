from pydantic import BaseModel, Field # type: ignore
from typing import Annotated, List
from typing_extensions import Literal

# Schema for structured output to use in planning
class Weapon(BaseModel):
    weapon_type: str = Field(
        description="Name of the weapon.",
    )
    weapon_name: str = Field(
        description="Special name given to the weapon.",
    )
    weapon_lore: str = Field(
        description="Lore behind the weaopon.",
    )

class SuperHero(BaseModel):
    name: str = Field(
        description="Formal name for the superhero.",
    )
    alias: str = Field(
        description="Hero name for the superhero.",
    )
    power: str = Field(
        description="Description of the power.",
    )
    origin: str = Field(
        description="City the superhero originiates from.",
    )
    weapon: Weapon = Field(
        description="This is the weapon of the superhero.",
    )


class Team(BaseModel):
    team_name: str = Field(
        description="Name of the superhero team.",
    )
    team: List[SuperHero] = Field(
        description="This is the team of superheroes.",
    )

class Story(BaseModel):
    title: str = Field(
        description="Title of the story.",
    )
    story: str = Field(
        description="This is the story of how the superhero team accomplishes the mission.",
    )

class StoryFeedback(BaseModel):
    grade: Literal["accepted", "not accepted"] = Field(
        description="Decide if the story is logical or not.",
    )
    feedback: str = Field(
        description="If the story is not logical, provide feedback on how to improve it.",
    )

class AnswerFeedback(BaseModel):
    grade: Literal["accepted", "not accepted"] = Field(
        description="Decide if the answer is logical or not.",
    )
    feedback: str = Field(
        description="If the answer is not logical, provide feedback on how to improve it.",
    )

class Decision(BaseModel):
    decision: List[Literal["ocr", "od", "sc"]] = Field(
        description="Decide which tools to use. ocr is for OCR, od is for object detection, and sc is for scene captioning.",
    )
    reasoning: str = Field(
        description="Provide reasoning for the decision.",
    )