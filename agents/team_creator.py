from langchain.schema import SystemMessage, HumanMessage
from prompts.system_message import TEAM_CREATOR_PROMPT_SYSTEM
from utils.logger import logger
from database.services import WeaponService, SuperHeroService, TeamService

class TeamGenerator:
    def __init__(self, llm):
        """Initialize the team generator agent."""
        self.llm = llm

    def generate_team(self, mission: str):
        """Generates a team of superheroes based on the mission."""
        logger.info(f"Generating team for mission: [yellow]{mission}[/yellow]")

        selected_team = self.llm.invoke([
            SystemMessage(content=TEAM_CREATOR_PROMPT_SYSTEM),
            HumanMessage(content=f"Here is the mission: {mission}")
        ])

        logger.success(f"Team Created: \"{selected_team.team_name}\" of {len(selected_team.team_name)} superheroes!")

        # Process and store team data
        self._store_team(selected_team)

        return {"team": selected_team.team, "team_name": selected_team.team_name}

    def _store_team(self, selected_team):
        """Stores the generated superheroes, weapons, and team in the database."""
        heroes_data = []

        for hero in selected_team.team:
            weapon_data = {
                "weapon_type": hero.weapon.weapon_type,
                "weapon_name": hero.weapon.weapon_name,
                "weapon_lore": hero.weapon.weapon_lore,
            }
            weapon_id = WeaponService.create_weapon(weapon_data)

            hero_data = {
                "name": hero.name,
                "alias": hero.alias,
                "power": hero.power,
                "origin": hero.origin,
                "weapon_id": weapon_id,  # Linking hero with the weapon
            }
            hero_id = SuperHeroService.create_superhero(hero_data)
            heroes_data.append(hero_id)

        
        team_data = {
            "team_name": selected_team.team_name,
            "hero_ids": heroes_data  # Storing the hero IDs
        }
        TeamService.create_team(team_data)

        logger.success(f"Stored {len(heroes_data)} heroes and their weapons in the database.")
