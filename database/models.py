class Weapon:
    def __init__(self, weapon_type, weapon_name, weapon_lore):
        self.weapon_type = weapon_type
        self.weapon_name = weapon_name
        self.weapon_lore = weapon_lore

    def to_dict(self):
        return {
            "weapon_type": self.weapon_type,
            "weapon_name": self.weapon_name,
            "weapon_lore": self.weapon_lore
        }

class SuperHero:
    def __init__(self, name, alias, power, origin, weapon):
        self.name = name
        self.alias = alias
        self.power = power
        self.origin = origin
        self.weapon = weapon  # Weapon should be a dictionary

    def to_dict(self):
        return {
            "name": self.name,
            "alias": self.alias,
            "power": self.power,
            "origin": self.origin,
            "weapon": self.weapon
        }

class Team:
    def __init__(self, team_name, superheroes):
        self.team_name = team_name
        self.superheroes = superheroes  # List of superhero dictionaries

    def to_dict(self):
        return {
            "team_name": self.team_name,
            "superheroes": self.superheroes
        }
