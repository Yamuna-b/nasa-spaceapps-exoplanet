"""
Database models and utilities for the authentication system.
"""
import sqlite3
import os
from datetime import datetime, timedelta
import uuid
from typing import Optional, Dict, Any
import json

class DatabaseManager:
    def __init__(self, db_path: str = "exoplanet_users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    bio TEXT,
                    institution TEXT,
                    research_interests TEXT,
                    is_verified BOOLEAN DEFAULT FALSE,
                    verification_token TEXT,
                    reset_token TEXT,
                    reset_token_expires TIMESTAMP,
                    google_id TEXT UNIQUE,
                    github_id TEXT UNIQUE,
                    profile_picture_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    preferences TEXT DEFAULT '{}'
                )
            """)
            
            # User sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                    user_id TEXT REFERENCES users(user_id),
                    session_token TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Login attempts table (for security)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS login_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    ip_address TEXT,
                    success BOOLEAN DEFAULT FALSE,
                    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def create_user(self, email: str, password_hash: str, name: str, 
                   verification_token: str = None) -> str:
        """Create a new user and return user_id."""
        user_id = str(uuid.uuid4())
        if not verification_token:
            verification_token = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (user_id, email, password_hash, name, verification_token)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, email.lower(), password_hash, name, verification_token))
            conn.commit()
        
        return user_id
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email address."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def verify_user_email(self, verification_token: str) -> bool:
        """Verify user email using verification token."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET is_verified = TRUE, verification_token = NULL 
                WHERE verification_token = ?
            """, (verification_token,))
            conn.commit()
            return cursor.rowcount > 0
    
    def update_user_profile(self, user_id: str, **kwargs) -> bool:
        """Update user profile information."""
        if not kwargs:
            return False
        
        # Build dynamic update query
        set_clauses = []
        values = []
        for key, value in kwargs.items():
            if key in ['name', 'bio', 'institution', 'research_interests', 'profile_picture_url']:
                set_clauses.append(f"{key} = ?")
                values.append(value)
        
        if not set_clauses:
            return False
        
        values.append(user_id)
        query = f"UPDATE users SET {', '.join(set_clauses)} WHERE user_id = ?"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
    
    def create_reset_token(self, email: str) -> Optional[str]:
        """Create password reset token for user."""
        reset_token = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=1)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET reset_token = ?, reset_token_expires = ?
                WHERE email = ?
            """, (reset_token, expires_at, email.lower()))
            conn.commit()
            
            if cursor.rowcount > 0:
                return reset_token
        return None
    
    def reset_password(self, reset_token: str, new_password_hash: str) -> bool:
        """Reset password using reset token."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET password_hash = ?, reset_token = NULL, reset_token_expires = NULL
                WHERE reset_token = ? AND reset_token_expires > CURRENT_TIMESTAMP
            """, (new_password_hash, reset_token))
            conn.commit()
            return cursor.rowcount > 0
    
    def update_last_login(self, user_id: str):
        """Update user's last login timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?
            """, (user_id,))
            conn.commit()
    
    def log_login_attempt(self, email: str, success: bool, ip_address: str = None):
        """Log login attempt for security monitoring."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO login_attempts (email, success, ip_address)
                VALUES (?, ?, ?)
            """, (email.lower(), success, ip_address))
            conn.commit()
    
    def get_recent_login_attempts(self, email: str, hours: int = 1) -> int:
        """Get number of recent failed login attempts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM login_attempts 
                WHERE email = ? AND success = FALSE AND attempted_at > ?
            """, (email.lower(), cutoff_time))
            return cursor.fetchone()[0]
    
    def create_session(self, user_id: str, session_token: str, expires_at: datetime) -> str:
        """Create user session."""
        session_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_sessions (session_id, user_id, session_token, expires_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, user_id, session_token, expires_at))
            conn.commit()
        
        return session_id
    
    def get_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get session by token."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM user_sessions 
                WHERE session_token = ? AND expires_at > CURRENT_TIMESTAMP
            """, (session_token,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def delete_session(self, session_token: str):
        """Delete user session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))
            conn.commit()
    
    def delete_all_user_sessions(self, user_id: str):
        """Delete all sessions for a user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
            conn.commit()

# Global database instance
db = DatabaseManager()
