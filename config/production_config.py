# config/production_config.py

class ProductionConfig:
    DEBUG = False  # Disable debug mode
    TESTING = False  # Disable testing mode
    DATABASE_URI = 'postgresql://user:password@localhost/prod_db'  # Replace with your production database URL
    SECRET_KEY = 'your_production_secret_key'  # Replace with your actual production secret key
    ALLOWED_HOSTS = ['yourdomain.com']  # Specify allowed hosts for production
    LOGGING_LEVEL = 'ERROR'  # Set logging level for production applications
    # Add any additional production-specific settings here