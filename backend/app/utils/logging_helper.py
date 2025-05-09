import os
import logging
from pathlib import Path

def setup_logging(log_file='app.log', level=logging.INFO):
    """
    Set up logging with file and console output.
    Ensures the log directory exists.
    
    Args:
        log_file: Name of the log file (will be created in logs directory)
        level: Logging level
    """
    # Get project root directory
    project_dir = Path(__file__).resolve().parent.parent.parent
    
    # Create logs directory if it doesn't exist
    logs_dir = project_dir / 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # Full path to log file
    log_path = logs_dir / log_file
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()