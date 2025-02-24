from .repositories import WeaponRepository, SuperHeroRepository, TeamRepository

class WeaponService:
    """Service layer for handling weapon operations."""

    @staticmethod
    def create_weapon(data):
        """Creates a new weapon in the database."""
        weapon = WeaponRepository.create_weapon(data)
        return weapon

    @staticmethod
    def get_weapon(weapon_id):
        """Retrieves a weapon by its ID."""
        print(f"🔍 Fetching weapon with ID: {weapon_id}")
        weapon = WeaponRepository.get_weapon(weapon_id)
        print(f"✅ Weapon found: {weapon}" if weapon else "❌ Weapon not found")
        return weapon


class SuperHeroService:
    @staticmethod
    def create_superhero(data):
        weapon = WeaponRepository.get_weapon(data["weapon_id"])
        if not weapon:
            return None  # Ensure the weapon exists before assigning it
        data["weapon"] = weapon
        superhero = SuperHeroRepository.create_superhero(data)
        return superhero

    @staticmethod
    def get_superhero(hero_id):
        return SuperHeroRepository.get_superhero(hero_id)

class TeamService:
    @staticmethod
    def create_team(data):
        """Creates a superhero team after ensuring all heroes exist in the database."""

        # Validate heroes
        heros = []
        for hero_id in data["hero_ids"]:
            hero = SuperHeroRepository.get_superhero(hero_id)
            if not hero:
                print(f"❌ Hero with ID {hero_id} not found! Skipping...")
                continue
            heros.append(hero)

        if len(heros) == 0:
            print("❌ No valid heroes found! Team creation aborted.")
            return None
        
        team_data = {
            "team_name": data["team_name"],
            "heros": heros  # Store hero references
        }

        team_id = TeamRepository.create_team(team_data)
        return team_id

    @staticmethod
    def get_team(team_id):
        """Retrieves a superhero team along with detailed hero data."""
        print(f"🔍 Fetching team with ID: {team_id}")
        team = TeamRepository.get_team(team_id)

        if not team:
            print(f"❌ Team with ID {team_id} not found!")
            return None

        # Fetch detailed hero data
        heroes = [SuperHeroRepository.get_superhero(hero_id) for hero_id in team["hero_ids"]]
        team["heroes"] = [hero for hero in heroes if hero]  # Remove None values if any hero wasn't found

        print(f"✅ Team \"{team['team_name']}\" retrieved successfully!")
        return team

