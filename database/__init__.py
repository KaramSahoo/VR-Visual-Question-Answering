from flask import g
from pymongo import MongoClient

def get_db():
    """Gets the database connection from Flask's global object g."""
    if 'db' not in g:
        client = MongoClient("mongodb://localhost:27017/")
        g.db = client.superhero_db  # Assign the database instance
    return g.db

def close_db(e=None):
    """Closes the database connection."""
    db = g.pop('db', None)
    if db:
        db.client.close()
