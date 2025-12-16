import torch
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"


DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

CSV_DATA_PATH = DATA_DIR / "final_chatbot_data.csv"


MODEL_NAME = "KETI-AIR/ke-t5-small"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = torch.cuda.is_available()

TRAIN_CONFIG = {
    "max_length": 128,
    "batch_size": 4,
    "learning_rate": 5e-4,
    "epochs": 4,
    "warmup_steps": 100,
    "logging_steps": 10,
}

TRAINING_CONFIG = {
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 100,
}

# Telegram
TELEGRAM_CONFIG = {
    "token": os.getenv("TELEGRAM_BOT_TOKEN", "ТВОЙ_ТОКЕН_ЕСЛИ_НЕТ_ENV")
}