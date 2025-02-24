import logging
from rich.logging import RichHandler

# Define log format with timestamp, level, module, and message
LOG_FORMAT = "[%(asctime)s] [purple]%(levelname)s[/purple] [%(module)s]: %(message)s"

# Custom success log level (between INFO and WARNING)
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def success(self, message, *args, **kwargs):
    """Custom success logging method with green color."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, f"[green]{message}[/green]", args, **kwargs)

# Add 'success' method to Logger class
logging.Logger.success = success

# Initialize logger
logger = logging.getLogger("workflow_logger")
logger.setLevel(logging.DEBUG)  # Capture all log levels

# Configure Rich Handler for colorful logs
console_handler = RichHandler(rich_tracebacks=True, markup=True)
formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)
