# run_seed.py
from database import get_memories

if __name__ == "__main__":
    success = get_memories()
    print(f"Seed operation {'succeeded' if success else 'failed'}")