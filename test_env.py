from dotenv import load_dotenv
from pathlib import Path
import os

dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)

print("🔍 All env vars with OPENAI in name:")
for k, v in os.environ.items():
    if "OPENAI" in k.upper():
        print(f"{k} = {v}")