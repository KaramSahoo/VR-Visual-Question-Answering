from flask import g
from pymongo import MongoClient
from utils.logger import logger
from pymongo.server_api import ServerApi

MONGO_URI = "mongodb+srv://ksahoo1:iODgPNH4hSEuGHEN@heroforge.eium3.mongodb.net/?retryWrites=true&w=majority&appName=HeroForge"
DB_NAME = "superhero_db"

def init_db():
    """Initialize the MongoDB connection and store it in g."""
    if 'db' not in g:
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

        # Send a ping to confirm a successful connection
        try:
            client.admin.command('ping')
            logger.info(f"You successfully connected to databases!")
        except Exception as e:
            logger.error(f"Error connecting to databases: {e}")

        g.db = client[DB_NAME]  # Store database reference in g

def get_db():
    """Get the database connection from g."""
    if 'db' not in g:
        init_db()
    return g.db

def close_db(e=None):
    """Closes the database connection."""
    db = g.pop('db', None)  # Remove db reference from g
    if db is not None:
        db.client.close()
