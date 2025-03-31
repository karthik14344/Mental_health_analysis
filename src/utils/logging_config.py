import logging
import os
from datetime import datetime

def setup_logging():
    """Configure logging for the application"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate log filename with timestamp
    log_filename = f"logs/mental_health_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # Reduce verbosity of external libraries
    logging.getLogger('gensim').setLevel(logging.WARNING)
    logging.getLogger('nltk').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)
