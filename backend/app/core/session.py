# app/core/session.py
import os
import uuid
import shutil
import sqlite3
import logging
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s: %(message)s',
#     handlers=[
#         logging.FileHandler('logs/session.log', mode='a'),
#         logging.StreamHandler()
#     ]
# )

class SessionManager:
    """Manages temporary user sessions for memory cartography application."""
    
    def __init__(self, base_temp_dir: str = None, cleanup_interval_hours: int = 24):
        """
        Initialize the session manager.
        
        Args:
            base_temp_dir: Root directory for temporary session storage
            cleanup_interval_hours: How long sessions persist before cleanup
        """
        # Create base temporary directory if not specified
        if base_temp_dir:
            self.base_temp_dir = Path(base_temp_dir)
        else:
            temp_root = Path(tempfile.gettempdir())
            self.base_temp_dir = temp_root / "memory_cartography_sessions"
        
        # Create the directory if it doesn't exist
        os.makedirs(self.base_temp_dir, exist_ok=True)
        
        # Setup session tracking
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = timedelta(hours=cleanup_interval_hours)
        
        # Initialize by loading any existing sessions
        self._load_existing_sessions()
        
        logging.info(f"Session manager initialized with base directory: {self.base_temp_dir}")
    
    def _load_existing_sessions(self):
        """Load information about existing session directories."""
        try:
            # Find all subdirectories in the base directory
            for session_dir in self.base_temp_dir.iterdir():
                if session_dir.is_dir():
                    session_id = session_dir.name
                    
                    # Check if it looks like a valid session directory
                    metadata_dir = session_dir / "metadata"
                    db_path = metadata_dir / "memories.db"
                    
                    if metadata_dir.exists() and db_path.exists():
                        # Load session info
                        info_path = metadata_dir / "session_info.json"
                        if info_path.exists():
                            try:
                                with open(info_path, 'r') as f:
                                    info = json.load(f)
                                    
                                # Convert string dates to datetime objects
                                for key in ['created_at', 'last_accessed']:
                                    if key in info and isinstance(info[key], str):
                                        info[key] = datetime.fromisoformat(info[key])
                                
                                # Register the session
                                self.sessions[session_id] = {
                                    "created_at": info.get('created_at', datetime.now()),
                                    "last_accessed": info.get('last_accessed', datetime.now()),
                                    "directory": str(session_dir),
                                    "db_path": str(db_path),
                                    "locations": set(info.get('locations', []))
                                }
                                logging.info(f"Loaded existing session: {session_id}")
                            except Exception as e:
                                logging.error(f"Error loading session info for {session_id}: {e}")
        except Exception as e:
            logging.error(f"Error loading existing sessions: {e}")
    
    def create_session(self) -> str:
        """
        Create a new user session with temporary directories.
        
        Returns:
            str: Unique session ID
        """
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Create session directory
        session_dir = self.base_temp_dir / session_id
        
        # Create directory structure
        os.makedirs(session_dir / "raw" / "user", exist_ok=True)
        os.makedirs(session_dir / "raw" / "public", exist_ok=True)
        os.makedirs(session_dir / "processed" / "user", exist_ok=True)
        os.makedirs(session_dir / "processed" / "public", exist_ok=True)
        os.makedirs(session_dir / "metadata", exist_ok=True)
        
        # Initialize SQLite database
        db_path = session_dir / "metadata" / "memories.db"
        self._init_db(db_path)
        
        # Record session information
        session_info = {
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "directory": str(session_dir),
            "db_path": str(db_path),
            "locations": set()
        }
        
        # Store in memory
        self.sessions[session_id] = session_info
        
        # Save session info to disk
        self._save_session_info(session_id)
        
        logging.info(f"Created new session: {session_id}")
        return session_id
    
    def _init_db(self, db_path: Path):
        """
        Initialize the SQLite database for a new session.
        
        Args:
            db_path: Path to the SQLite database file
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create memories table with all needed columns
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            location TEXT NOT NULL,
            date TEXT NOT NULL,
            type TEXT NOT NULL,
            keywords TEXT,
            description TEXT,
            filename TEXT,
            weight REAL DEFAULT 1.0,
            embedding TEXT,
            openai_keywords TEXT,
            openai_description TEXT,
            impact_weight REAL DEFAULT 1.0,
            resnet_embedding TEXT,
            detected_objects TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _save_session_info(self, session_id: str):
        """
        Save session information to disk.
        
        Args:
            session_id: Unique session identifier
        """
        if session_id not in self.sessions:
            return
        
        try:
            session_info = self.sessions[session_id].copy()
            
            # Convert datetime objects to ISO format strings
            for key in ['created_at', 'last_accessed']:
                if key in session_info and isinstance(session_info[key], datetime):
                    session_info[key] = session_info[key].isoformat()
            
            # Convert set to list for JSON serialization
            if 'locations' in session_info and isinstance(session_info['locations'], set):
                session_info['locations'] = list(session_info['locations'])
            
            # Save to file
            metadata_dir = Path(session_info['directory']) / "metadata"
            info_path = metadata_dir / "session_info.json"
            
            with open(info_path, 'w') as f:
                json.dump(session_info, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving session info for {session_id}: {e}")
    
    def get_session_paths(self, session_id: str) -> Optional[Dict[str, str]]:
        """
        Get all relevant paths for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Dict with paths or None if session not found
        """
        if session_id not in self.sessions:
            return None
        
        # Update last accessed time
        self.sessions[session_id]["last_accessed"] = datetime.now()
        self._save_session_info(session_id)
        
        session_dir = self.sessions[session_id]["directory"]
        
        return {
            "root": session_dir,
            "raw_user": os.path.join(session_dir, "raw", "user"),
            "raw_public": os.path.join(session_dir, "raw", "public"),
            "processed_user": os.path.join(session_dir, "processed", "user"),
            "processed_public": os.path.join(session_dir, "processed", "public"),
            "metadata": os.path.join(session_dir, "metadata"),
            "db_path": self.sessions[session_id]["db_path"]
        }
    
    def add_location(self, session_id: str, location: str):
        """
        Add a detected location to the session.
        
        Args:
            session_id: Unique session identifier
            location: Location name to add
        """
        if session_id in self.sessions and location:
            if location.lower() != "unknown location":
                # Extract main part of location before first comma
                main_location = location.split(',')[0].strip()
                self.sessions[session_id]["locations"].add(main_location)
                # Save updated info
                self._save_session_info(session_id)
    
    def get_locations(self, session_id: str) -> List[str]:
        """
        Get all detected locations for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            List of location names
        """
        if session_id in self.sessions:
            return list(self.sessions[session_id]["locations"])
        return []
    
    def cleanup_expired_sessions(self):
        """
        Remove sessions that haven't been accessed recently.
        """
        now = datetime.now()
        expired_sessions = []
        
        for session_id, info in self.sessions.items():
            if now - info["last_accessed"] > self.cleanup_interval:
                try:
                    # Remove session directory
                    shutil.rmtree(info["directory"])
                    expired_sessions.append(session_id)
                    logging.info(f"Cleaned up expired session: {session_id}")
                except Exception as e:
                    logging.error(f"Error cleaning up session {session_id}: {e}")
        
        # Remove expired sessions from memory
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Explicitly delete a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if session was deleted, False otherwise
        """
        if session_id not in self.sessions:
            return False
        
        try:
            # Remove session directory
            shutil.rmtree(self.sessions[session_id]["directory"])
            # Remove from memory
            del self.sessions[session_id]
            logging.info(f"Deleted session: {session_id}")
            return True
        except Exception as e:
            logging.error(f"Error deleting session {session_id}: {e}")
            return False

# Create a singleton instance
session_manager = SessionManager()

def get_session_manager() -> SessionManager:
    """Get the singleton instance of SessionManager."""
    return session_manager