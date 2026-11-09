"""Initialize the SQLite database tables."""
from dotenv import load_dotenv
load_dotenv()
from backend.database import init_db
init_db()
print("Database initialized successfully.")
