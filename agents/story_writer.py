from langchain.schema import SystemMessage, HumanMessage
from utils.logger import logger
from prompts.system_message import STORY_PROMPT_SYSTEM, IMPROVE_STORY_PROMPT_SYSTEM
from prompts.user_prompts import STORY_PROMPT_USER, IMPROVE_STORY_PROMPT_USER


class StoryGenerator:
    def __init__(self, llm):
        """Initialize the story generator agent."""
        self.llm = llm

    def generate_story_prompt(self, mission: str, team_name: str, heroes):
        """Formats the story prompt with mission and hero details."""
        heroes_details = "\n".join([
            f"- {hero.name} ({hero.alias}): {hero.power}" 
            for hero in heroes
        ])

        return STORY_PROMPT_USER.format(
            mission=mission,
            team_name=team_name,
            heroes_details=heroes_details or "No heroes assigned yet."
        )
    
    def improve_story_prompt(self, mission: str, team_name: str, heroes, feedback: str, story: str):
        """Formats the story improvement prompt with mission, hero and feedback details."""
        heroes_details = "\n".join([
            f"- {hero.name} ({hero.alias}): {hero.power}" 
            for hero in heroes
        ])

        return IMPROVE_STORY_PROMPT_USER.format(
            mission=mission,
            team_name=team_name,
            story=story,
            feedback=feedback,
            heroes_details=heroes_details or "No heroes assigned yet."
        )

    def generate_story(self, mission: str, team_name: str, heroes):
        """Generates a story for the mission and superheroes."""
        logger.info("Generating a mission story...")
        story_prompt_user = self.generate_story_prompt(mission, team_name, heroes)

        mission_story = self.llm.invoke([
            SystemMessage(content=STORY_PROMPT_SYSTEM),
            HumanMessage(content=story_prompt_user)
        ])

        logger.success(f"[green]Story generated successfully with title: {mission_story.title}![/green]")

        return {"story": mission_story.story}
    
    def improve_story(self, mission: str, team_name: str, heroes, feedback: str, story: str):
        """Improves the story based on feedback."""
        logger.info("Improving the story based on feedback...")
        improve_story_prompt_user = self.improve_story_prompt(mission, team_name, heroes, feedback, story)

        mission_story = self.llm.invoke([
            SystemMessage(content=IMPROVE_STORY_PROMPT_SYSTEM),
            HumanMessage(content=improve_story_prompt_user)
        ])

        logger.success(f"[green]Story improved successfully![/green]")

        return {"story": mission_story.story}
