import os
from dotenv import load_dotenv

load_dotenv()

print("Anthropic key loaded:", bool(os.getenv("ANTHROPIC_API_KEY")))
print("Voyage key loaded:", bool(os.getenv("VOYAGE_API_KEY")))
