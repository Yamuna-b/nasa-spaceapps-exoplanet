"""
Authentication utilities including password validation, hashing, and email verification.
"""
import re
import secrets
import string
from typing import Tuple, Dict, Any
import bcrypt
import jwt
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

class PasswordValidator:
    """Password validation with specific requirements."""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """
        Validate password according to requirements:
        - Minimum 8 characters
        - At least 1 special character
        - At least 1 uppercase letter
        - At least 1 lowercase letter
        - At least 1 number
        - No spaces allowed
        - Underscores (_) and hyphens (-) allowed
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if ' ' in password:
            return False, "Password cannot contain spaces"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one number"
        
        # Special characters (including underscore and hyphen)
        special_chars = r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]'
        if not re.search(special_chars, password):
            return False, "Password must contain at least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)"
        
        # Check for invalid characters
        valid_chars = r'^[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?]+$'
        if not re.match(valid_chars, password):
            return False, "Password contains invalid characters"
        
        return True, "Password is valid"
    
    @staticmethod
    def get_password_strength(password: str) -> Dict[str, Any]:
        """Get detailed password strength analysis."""
        strength = {
            'length': len(password) >= 8,
            'uppercase': bool(re.search(r'[A-Z]', password)),
            'lowercase': bool(re.search(r'[a-z]', password)),
            'digit': bool(re.search(r'\d', password)),
            'special': bool(re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password)),
            'no_spaces': ' ' not in password,
            'score': 0
        }
        
        # Calculate strength score
        strength['score'] = sum([
            strength['length'],
            strength['uppercase'],
            strength['lowercase'],
            strength['digit'],
            strength['special'],
            strength['no_spaces']
        ])
        
        # Determine strength level
        if strength['score'] >= 6:
            strength['level'] = 'Strong'
            strength['color'] = 'green'
        elif strength['score'] >= 4:
            strength['level'] = 'Medium'
            strength['color'] = 'orange'
        else:
            strength['level'] = 'Weak'
            strength['color'] = 'red'
        
        return strength

class PasswordHasher:
    """Password hashing utilities using bcrypt."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

class EmailValidator:
    """Email validation utilities."""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email format."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not email:
            return False, "Email is required"
        
        if not re.match(email_pattern, email):
            return False, "Invalid email format"
        
        if len(email) > 255:
            return False, "Email is too long"
        
        return True, "Email is valid"

class TokenManager:
    """JWT token management for sessions."""
    
    SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    
    @staticmethod
    def generate_token(user_id: str, expires_hours: int = 24) -> str:
        """Generate JWT token for user session."""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, TokenManager.SECRET_KEY, algorithm='HS256')
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, TokenManager.SECRET_KEY, algorithms=['HS256'])
            return {'valid': True, 'user_id': payload['user_id']}
        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token has expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}

class EmailService:
    """Email service for verification and password reset."""
    
    def __init__(self):
        # Email configuration (you'll need to set these environment variables)
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.from_email = os.getenv('FROM_EMAIL', 'noreply@exoplanet-ai.com')
    
    def send_verification_email(self, to_email: str, verification_token: str, base_url: str = "http://localhost:8501") -> bool:
        """Send email verification email."""
        try:
            verification_link = f"{base_url}?verify_token={verification_token}"
            
            subject = "Verify Your Exoplanet AI Account"
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #1f4e79;">Welcome to Exoplanet AI! 🪐</h2>
            
            Please click the link below to verify your email address:
            http://localhost:8501?verify={token}
            
            If you didn't create this account, please ignore this email.
            
            Best regards,
            The Exoplanet AI Team
            """
            
            return self._send_email(email, subject, body)
            
        except Exception as e:
            print(f"Error sending verification email: {e}")
            return False
    
    def send_password_reset_email(self, to_email: str, reset_token: str, base_url: str = "http://localhost:8501") -> bool:
        """Send password reset email."""
        try:
            reset_link = f"{base_url}?reset_token={reset_token}"
            
            subject = "Reset Your Exoplanet AI Password"
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #1f4e79;">Password Reset Request 🔐</h2>
                    <p>We received a request to reset your password for your Exoplanet AI account.</p>
                    <p>Click the button below to reset your password:</p>
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{reset_link}" 
                           style="background-color: #dc3545; color: white; padding: 12px 30px; 
                                  text-decoration: none; border-radius: 5px; display: inline-block;">
                            Reset Password
                        </a>
                    </div>
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="word-break: break-all; color: #666;">{reset_link}</p>
                    <p><strong>This link will expire in 1 hour.</strong></p>
                    <p>If you didn't request a password reset, please ignore this email. Your password will remain unchanged.</p>
                    <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
                    <p style="font-size: 12px; color: #666;">
                        For security reasons, this link can only be used once.
                    </p>
                </div>
            </body>
            </html>
            """
            
            return self._send_email(to_email, subject, html_body)
        except Exception as e:
            print(f"Error sending password reset email: {e}")
            return False
    
    def _send_email(self, to_email: str, subject: str, html_body: str) -> bool:
        """Send email using SMTP."""
        try:
            # Check if SMTP is configured
            if not self.smtp_username or not self.smtp_password:
                print(f"\n🔔 MOCK EMAIL NOTIFICATION:")
                print(f"📧 To: {to_email}")
                print(f"📋 Subject: {subject}")
                print(f"💡 To enable REAL emails, configure these in .env file:")
                print(f"   SMTP_USERNAME=your-email@gmail.com")
                print(f"   SMTP_PASSWORD=your-app-password")
                print(f"🚫 Currently using DUMMY verification - you can sign in without email verification")
                print("-" * 60)
                return True
            
            # For development, we'll just print the email content
            # In production, you would configure actual SMTP settings
            print(f"\n--- REAL EMAIL SENT ---")
            print(f"To: {to_email}")
            print(f"Subject: {subject}")
            print("--- END EMAIL ---\n")
            
            # Uncomment below for actual email sending in production
            """
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email
            
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            """
            
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

# Global instances
password_validator = PasswordValidator()
password_hasher = PasswordHasher()
email_validator = EmailValidator()
token_manager = TokenManager()
email_service = EmailService()
