# config/__init__.py

import os
from .development_config import DevelopmentConfig
from .production_config import ProductionConfig

def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    if env == 'production':
        return ProductionConfig()
    return DevelopmentConfig()

config = get_config()