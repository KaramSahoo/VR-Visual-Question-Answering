from bson.objectid import ObjectId
from .connection import get_db

# --- Weapon Repository ---
class WeaponRepository:
    @staticmethod
    def create_weapon(data):
        return get_db().weapons.insert_one(data).inserted_id

    @staticmethod
    def get_weapon(weapon_id):
        return get_db().weapons.find_one({"_id": ObjectId(weapon_id)})

    @staticmethod
    def update_weapon(weapon_id, data):
        return get_db().weapons.update_one({"_id": ObjectId(weapon_id)}, {"$set": data})

    @staticmethod
    def delete_weapon(weapon_id):
        return get_db().weapons.delete_one({"_id": ObjectId(weapon_id)})

# --- SuperHero Repository ---
class SuperHeroRepository:
    @staticmethod
    def create_superhero(data):
        return get_db().superheroes.insert_one(data).inserted_id

    @staticmethod
    def get_superhero(hero_id):
        return get_db().superheroes.find_one({"_id": ObjectId(hero_id)})

    @staticmethod
    def update_superhero(hero_id, data):
        return get_db().superheroes.update_one({"_id": ObjectId(hero_id)}, {"$set": data})

    @staticmethod
    def delete_superhero(hero_id):
        return get_db().superheroes.delete_one({"_id": ObjectId(hero_id)})
    
from bson.objectid import ObjectId
from .connection import get_db

# --- Team Repository ---
class TeamRepository:
    @staticmethod
    def create_team(data):
        team_id = get_db().teams.insert_one(data).inserted_id
        return team_id

    @staticmethod
    def get_team(team_id):
        team = get_db().teams.find_one({"_id": ObjectId(team_id)})
        return team

    @staticmethod
    def update_team(team_id, data):
        result = get_db().teams.update_one({"_id": ObjectId(team_id)}, {"$set": data})
        return result

    @staticmethod
    def delete_team(team_id):
        result = get_db().teams.delete_one({"_id": ObjectId(team_id)})
        return result.deleted_count > 0

