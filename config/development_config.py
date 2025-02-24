class DevelopmentConfig:
    DEBUG = True  # Enable debug mode
    TESTING = True  # Enable testing mode
    DATABASE_URI = 'sqlite:///:memory:'  # Use an in-memory SQLite database for development
    SECRET_KEY = 'your_development_secret_key'  # Replace with your actual development secret key
    ALLOWED_HOSTS = ['*']  # Allow localhost for development
    PORT = '8000' # Set port number for allowed hosts
    LOGGING_LEVEL = 'DEBUG'  # Set logging level for development applications
    # Add any additional development-specific settings here