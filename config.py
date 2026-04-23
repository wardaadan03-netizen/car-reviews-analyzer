import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    # Model settings
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo')
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '1000'))
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    
    # Analysis settings
    SENTIMENT_THRESHOLDS = {
        'positive': 0.6,
        'neutral': 0.4,
        'negative': 0.2
    }
    
    # Car aspects to analyze
    CAR_ASPECTS = [
        'performance', 'comfort', 'fuel_economy', 'safety',
        'reliability', 'design', 'technology', 'value_for_money',
        'maintenance_cost', 'interior_quality'
    ]
    
    # Create directories if they don't exist
    @classmethod
    def setup_directories(cls):
        for dir_path in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

config = Config()