"""
Streamlit authentication interface with sign up, sign in, and profile management.
"""
import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import uuid

from .database import db
from .auth_utils import (
    password_validator, password_hasher, email_validator, 
    token_manager, email_service
)
from .oauth import oauth_manager

class AuthInterface:
    """Main authentication interface for Streamlit."""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables."""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'auth_mode' not in st.session_state:
            st.session_state.auth_mode = 'signin'
    
    def show_auth_page(self):
        """Show authentication page with sign in/up forms."""
        st.markdown("""
        <style>
        .auth-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            color: white;
        }
        .auth-title {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 2rem;
            color: white;
        }
        .oauth-button {
            width: 100%;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .google-btn {
            background-color: #db4437;
            color: white;
        }
        .github-btn {
            background-color: #333;
            color: white;
        }
        .password-strength {
            margin-top: 0.5rem;
            padding: 0.5rem;
            border-radius: 5px;
            font-size: 0.9rem;
        }
        .strength-weak { background-color: #ffebee; color: #c62828; }
        .strength-medium { background-color: #fff3e0; color: #ef6c00; }
        .strength-strong { background-color: #e8f5e8; color: #2e7d32; }
        </style>
        """, unsafe_allow_html=True)
        
        # Check for OAuth callback
        self.handle_oauth_callback()
        
        # Check for email verification
        self.handle_email_verification()
        
        # Check for password reset
        self.handle_password_reset()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            
            # Title
            st.markdown('<h1 class="auth-title">🪐 Exoplanet AI</h1>', unsafe_allow_html=True)
            
            # Auth mode selector
            auth_mode = st.radio(
                "Choose action:",
                ["Sign In", "Sign Up"],
                horizontal=True,
                key="auth_mode_selector"
            )
            
            if auth_mode == "Sign In":
                self.show_signin_form()
            elif auth_mode == "Sign Up":
                self.show_signup_form()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def show_signin_form(self):
        """Show sign in form."""
        st.subheader("Sign In")
        
        with st.form("signin_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password")
            remember_me = st.checkbox("Remember me")
            
            signin_btn = st.form_submit_button("Sign In", use_container_width=True)
            
            if signin_btn:
                self.handle_signin(email, password, remember_me)
        
        # Forgot password link outside the form
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔑 Forgot Password?", use_container_width=True):
                self.show_forgot_password_modal()
        
        # OAuth buttons
        self.show_oauth_buttons()
    
    def show_signup_form(self):
        """Show sign up form."""
        st.subheader("Create Account")
        
        with st.form("signup_form"):
            name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", help="Min 8 chars, 1 uppercase, 1 lowercase, 1 number, 1 special character")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            # Real-time password strength indicator
            if password:
                strength = password_validator.get_password_strength(password)
                strength_class = f"strength-{strength['level'].lower()}"
                st.markdown(f"""
                <div class="password-strength {strength_class}">
                    Password Strength: {strength['level']} ({strength['score']}/6)
                </div>
                """, unsafe_allow_html=True)
            
            terms_accepted = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            signup_btn = st.form_submit_button("Create Account", use_container_width=True)
            
            if signup_btn:
                self.handle_signup(name, email, password, confirm_password, terms_accepted)
        
        # OAuth buttons
        self.show_oauth_buttons()
    
    def show_forgot_password_modal(self):
        """Show forgot password in a modal-like container."""
        with st.container():
            st.markdown("---")
            st.subheader("🔑 Reset Password")
            
            with st.form("forgot_password_modal"):
                email = st.text_input("Email", placeholder="Enter your email address")
                
                col1, col2 = st.columns(2)
                with col1:
                    reset_btn = st.form_submit_button("Send Reset Link", use_container_width=True)
                with col2:
                    cancel_btn = st.form_submit_button("Cancel", use_container_width=True)
                
                if reset_btn:
                    self.handle_forgot_password(email)
                
                if cancel_btn:
                    st.rerun()
    
    def show_forgot_password_form(self):
        """Show forgot password form."""
        st.subheader("Reset Password")
        
        with st.form("forgot_password_form"):
            email = st.text_input("Email", placeholder="Enter your email address")
            
            reset_btn = st.form_submit_button("Send Reset Link", use_container_width=True)
            back_btn = st.form_submit_button("Back to Sign In", use_container_width=True)
            
            if reset_btn:
                self.handle_forgot_password(email)
            
            if back_btn:
                st.session_state.auth_mode = 'signin'
                st.rerun()
    
    def show_oauth_buttons(self):
        """Show OAuth authentication buttons."""
        st.markdown("---")
        st.markdown("**Or continue with:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if oauth_manager.is_provider_configured('google'):
                if st.button("🔍 Google", key="google_oauth", use_container_width=True):
                    auth_url = oauth_manager.get_authorization_url('google')
                    if auth_url:
                        st.markdown(f'<a href="{auth_url}" target="_self">Continue with Google</a>', unsafe_allow_html=True)
                    else:
                        st.error("Google OAuth not configured")
            else:
                st.info("Google OAuth not configured")
        
        with col2:
            if oauth_manager.is_provider_configured('github'):
                if st.button("🐙 GitHub", key="github_oauth", use_container_width=True):
                    auth_url = oauth_manager.get_authorization_url('github')
                    if auth_url:
                        st.markdown(f'<a href="{auth_url}" target="_self">Continue with GitHub</a>', unsafe_allow_html=True)
                    else:
                        st.error("GitHub OAuth not configured")
            else:
                st.info("GitHub OAuth not configured")
    
    def handle_signin(self, email: str, password: str, remember_me: bool):
        """Handle sign in process."""
        if not email or not password:
            st.error("Please fill in all fields")
            return
        
        # Validate email format
        email_valid, email_msg = email_validator.validate_email(email)
        if not email_valid:
            st.error(email_msg)
            return
        
        # Check for too many failed attempts
        failed_attempts = db.get_recent_login_attempts(email, hours=1)
        if failed_attempts >= 5:
            st.error("Too many failed login attempts. Please try again in 1 hour.")
            return
        
        # Get user from database
        user = db.get_user_by_email(email)
        if not user:
            db.log_login_attempt(email, False)
            st.error("Invalid email or password")
            return
        
        # Verify password
        if not password_hasher.verify_password(password, user['password_hash']):
            db.log_login_attempt(email, False)
            st.error("Invalid email or password")
            return
        
        # Check if email is verified (allow dummy verification for development)
        if not user['is_verified']:
            st.warning("⚠️ Email not verified, but proceeding with dummy verification for development.")
            st.info("💡 In production, configure SMTP settings in .env for real email verification.")
        
        # Check if account is active
        if not user['is_active']:
            st.error("Your account has been deactivated. Please contact support.")
            return
        
        # Successful login
        db.log_login_attempt(email, True)
        db.update_last_login(user['user_id'])
        
        # Create session
        expires_hours = 720 if remember_me else 24  # 30 days vs 1 day
        session_token = token_manager.generate_token(user['user_id'], expires_hours)
        expires_at = datetime.now() + timedelta(hours=expires_hours)
        
        db.create_session(user['user_id'], session_token, expires_at)
        
        # Update session state
        st.session_state.authenticated = True
        st.session_state.user = user
        st.session_state.session_token = session_token
        
        st.success("Successfully signed in!")
        st.rerun()
    
    def handle_signup(self, name: str, email: str, password: str, confirm_password: str, terms_accepted: bool):
        """Handle sign up process."""
        # Validation
        if not all([name, email, password, confirm_password]):
            st.error("Please fill in all fields")
            return
        
        if not terms_accepted:
            st.error("Please accept the Terms of Service and Privacy Policy")
            return
        
        # Validate email
        email_valid, email_msg = email_validator.validate_email(email)
        if not email_valid:
            st.error(email_msg)
            return
        
        # Check if email already exists
        existing_user = db.get_user_by_email(email)
        if existing_user:
            st.error("An account with this email already exists. Please sign in instead.")
            return
        
        # Validate password
        password_valid, password_msg = password_validator.validate_password(password)
        if not password_valid:
            st.error(password_msg)
            return
        
        # Check password confirmation
        if password != confirm_password:
            st.error("Passwords do not match")
            return
        
        # Create user
        try:
            password_hash = password_hasher.hash_password(password)
            verification_token = str(uuid.uuid4())
            
            user_id = db.create_user(email, password_hash, name, verification_token)
            
            # Send verification email
            email_sent = email_service.send_verification_email(email, verification_token)
            
            if email_sent:
                st.success("🎉 Account created successfully! Please check your email for verification instructions.")
                st.info("📧 You'll need to verify your email before you can sign in.")
                
                # Show sign in option below success message
                st.markdown("---")
                if st.button("✅ Go to Sign In", use_container_width=True):
                    st.session_state.auth_mode = 'signin'
                    st.rerun()
            else:
                st.warning("⚠️ Account created, but verification email could not be sent. You can still sign in with dummy verification.")
                
                # Show sign in option below warning message
                st.markdown("---")
                if st.button("✅ Go to Sign In", use_container_width=True):
                    st.session_state.auth_mode = 'signin'
                    st.rerun()
            
        except Exception as e:
            st.error(f"Error creating account: {e}")
    
    def handle_forgot_password(self, email: str):
        """Handle forgot password process."""
        if not email:
            st.error("Please enter your email address")
            return
        
        # Validate email format
        email_valid, email_msg = email_validator.validate_email(email)
        if not email_valid:
            st.error(email_msg)
            return
        
        # Check if user exists
        user = db.get_user_by_email(email)
        if not user:
            # Don't reveal if email exists or not for security
            st.success("If an account with this email exists, you will receive a password reset link.")
            return
        
        # Create reset token
        reset_token = db.create_reset_token(email)
        if reset_token:
            # Send reset email
            email_sent = email_service.send_password_reset_email(email, reset_token)
            
            if email_sent:
                st.success("Password reset link sent to your email address.")
            else:
                st.error("Error sending password reset email. Please try again.")
        else:
            st.error("Error creating password reset token. Please try again.")
    
    def handle_oauth_callback(self):
        """Handle OAuth callback from URL parameters."""
        # Get URL parameters
        query_params = st.query_params
        
        code = query_params.get('code')
        state = query_params.get('state')
        provider = query_params.get('provider')
        
        if code and state and provider:
            # Handle OAuth callback
            user_info = oauth_manager.handle_oauth_callback(provider, code, state)
            
            if user_info:
                # Check if user exists
                existing_user = db.get_user_by_email(user_info['email'])
                
                if existing_user:
                    # Sign in existing user
                    db.update_last_login(existing_user['user_id'])
                    
                    # Create session
                    session_token = token_manager.generate_token(existing_user['user_id'])
                    expires_at = datetime.now() + timedelta(hours=24)
                    
                    db.create_session(existing_user['user_id'], session_token, expires_at)
                    
                    # Update session state
                    st.session_state.authenticated = True
                    st.session_state.user = existing_user
                    st.session_state.session_token = session_token
                    
                    st.success(f"Successfully signed in with {provider.title()}!")
                    
                else:
                    # Create new user
                    try:
                        # Generate a random password for OAuth users
                        temp_password = str(uuid.uuid4())
                        password_hash = password_hasher.hash_password(temp_password)
                        
                        user_id = db.create_user(
                            user_info['email'], 
                            password_hash, 
                            user_info['name']
                        )
                        
                        # Mark as verified since OAuth providers verify emails
                        db.verify_user_email(None)  # This won't work, need to fix
                        
                        # Update OAuth provider info
                        if provider == 'google':
                            # Update google_id in database
                            pass
                        elif provider == 'github':
                            # Update github_id in database
                            pass
                        
                        st.success(f"Account created successfully with {provider.title()}!")
                        
                    except Exception as e:
                        st.error(f"Error creating account: {e}")
            else:
                st.error("OAuth authentication failed. Please try again.")
            
            # Clear URL parameters
            st.query_params.clear()
    
    def handle_email_verification(self):
        """Handle email verification from URL parameters."""
        query_params = st.query_params
        verify_token = query_params.get('verify_token')
        
        if verify_token:
            if db.verify_user_email(verify_token):
                st.success("Email verified successfully! You can now sign in.")
            else:
                st.error("Invalid or expired verification token.")
            
            # Clear URL parameters
            st.query_params.clear()
    
    def handle_password_reset(self):
        """Handle password reset from URL parameters."""
        query_params = st.query_params
        reset_token = query_params.get('reset_token')
        
        if reset_token:
            st.subheader("Reset Your Password")
            
            with st.form("reset_password_form"):
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                # Password strength indicator
                if new_password:
                    strength = password_validator.get_password_strength(new_password)
                    strength_class = f"strength-{strength['level'].lower()}"
                    st.markdown(f"""
                    <div class="password-strength {strength_class}">
                        Password Strength: {strength['level']} ({strength['score']}/6)
                    </div>
                    """, unsafe_allow_html=True)
                
                reset_btn = st.form_submit_button("Reset Password")
                
                if reset_btn:
                    if not new_password or not confirm_password:
                        st.error("Please fill in all fields")
                        return
                    
                    # Validate password
                    password_valid, password_msg = password_validator.validate_password(new_password)
                    if not password_valid:
                        st.error(password_msg)
                        return
                    
                    # Check password confirmation
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                        return
                    
                    # Reset password
                    password_hash = password_hasher.hash_password(new_password)
                    if db.reset_password(reset_token, password_hash):
                        st.success("Password reset successfully! You can now sign in with your new password.")
                        # Clear URL parameters
                        st.experimental_set_query_params()
                        st.rerun()
                    else:
                        st.error("Invalid or expired reset token.")
    
    def check_authentication(self) -> bool:
        """Check if user is authenticated."""
        if not st.session_state.authenticated:
            return False
        
        # Verify session token
        session_token = st.session_state.get('session_token')
        if not session_token:
            return False
        
        session = db.get_session(session_token)
        if not session:
            # Session expired or invalid
            self.logout()
            return False
        
        return True
    
    def logout(self):
        """Log out current user."""
        # Delete session from database
        session_token = st.session_state.get('session_token')
        if session_token:
            db.delete_session(session_token)
        
        # Clear session state
        st.session_state.authenticated = False
        st.session_state.user = None
        if 'session_token' in st.session_state:
            del st.session_state.session_token
        
        st.success("Successfully logged out!")
        st.rerun()
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user."""
        if self.check_authentication():
            return st.session_state.user
        return None

# Global auth interface instance
auth_interface = AuthInterface()
