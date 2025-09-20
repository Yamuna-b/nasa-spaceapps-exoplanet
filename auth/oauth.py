"""
OAuth integration for Google and GitHub authentication.
"""
import os
import secrets
import httpx
from typing import Dict, Any, Optional
from urllib.parse import urlencode
import streamlit as st

class OAuthProvider:
    """Base OAuth provider class."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
    
    def get_authorization_url(self, state: str) -> str:
        """Get OAuth authorization URL."""
        raise NotImplementedError
    
    def exchange_code_for_token(self, code: str, state: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token."""
        raise NotImplementedError
    
    def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user information using access token."""
        raise NotImplementedError

class GoogleOAuth(OAuthProvider):
    """Google OAuth provider."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        super().__init__(client_id, client_secret, redirect_uri)
        self.auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        self.user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        self.scope = "openid email profile"
    
    def get_authorization_url(self, state: str) -> str:
        """Get Google OAuth authorization URL."""
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': self.scope,
            'response_type': 'code',
            'state': state,
            'access_type': 'offline',
            'prompt': 'consent'
        }
        return f"{self.auth_url}?{urlencode(params)}"
    
    def exchange_code_for_token(self, code: str, state: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for Google access token."""
        try:
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': self.redirect_uri
            }
            
            with httpx.Client() as client:
                response = client.post(self.token_url, data=data)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Error exchanging Google code for token: {e}")
            return None
    
    def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get Google user information."""
        try:
            headers = {'Authorization': f'Bearer {access_token}'}
            
            with httpx.Client() as client:
                response = client.get(self.user_info_url, headers=headers)
                response.raise_for_status()
                user_data = response.json()
                
                # Normalize user data
                return {
                    'id': user_data.get('id'),
                    'email': user_data.get('email'),
                    'name': user_data.get('name'),
                    'picture': user_data.get('picture'),
                    'verified_email': user_data.get('verified_email', False),
                    'provider': 'google'
                }
        except Exception as e:
            print(f"Error getting Google user info: {e}")
            return None

class GitHubOAuth(OAuthProvider):
    """GitHub OAuth provider."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        super().__init__(client_id, client_secret, redirect_uri)
        self.auth_url = "https://github.com/login/oauth/authorize"
        self.token_url = "https://github.com/login/oauth/access_token"
        self.user_info_url = "https://api.github.com/user"
        self.user_email_url = "https://api.github.com/user/emails"
        self.scope = "user:email"
    
    def get_authorization_url(self, state: str) -> str:
        """Get GitHub OAuth authorization URL."""
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': self.scope,
            'state': state
        }
        return f"{self.auth_url}?{urlencode(params)}"
    
    def exchange_code_for_token(self, code: str, state: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for GitHub access token."""
        try:
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code
            }
            
            headers = {'Accept': 'application/json'}
            
            with httpx.Client() as client:
                response = client.post(self.token_url, data=data, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Error exchanging GitHub code for token: {e}")
            return None
    
    def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get GitHub user information."""
        try:
            headers = {
                'Authorization': f'token {access_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            with httpx.Client() as client:
                # Get user profile
                user_response = client.get(self.user_info_url, headers=headers)
                user_response.raise_for_status()
                user_data = user_response.json()
                
                # Get user emails
                email_response = client.get(self.user_email_url, headers=headers)
                email_response.raise_for_status()
                emails = email_response.json()
                
                # Find primary email
                primary_email = None
                for email in emails:
                    if email.get('primary', False):
                        primary_email = email.get('email')
                        break
                
                # Normalize user data
                return {
                    'id': str(user_data.get('id')),
                    'email': primary_email or user_data.get('email'),
                    'name': user_data.get('name') or user_data.get('login'),
                    'picture': user_data.get('avatar_url'),
                    'verified_email': True,  # GitHub emails are considered verified
                    'provider': 'github',
                    'username': user_data.get('login'),
                    'bio': user_data.get('bio'),
                    'company': user_data.get('company'),
                    'location': user_data.get('location'),
                    'blog': user_data.get('blog')
                }
        except Exception as e:
            print(f"Error getting GitHub user info: {e}")
            return None

class OAuthManager:
    """OAuth manager for handling multiple providers."""
    
    def __init__(self):
        # OAuth configuration from environment variables
        self.google_client_id = os.getenv('GOOGLE_CLIENT_ID', '')
        self.google_client_secret = os.getenv('GOOGLE_CLIENT_SECRET', '')
        self.github_client_id = os.getenv('GITHUB_CLIENT_ID', '')
        self.github_client_secret = os.getenv('GITHUB_CLIENT_SECRET', '')
        
        # Redirect URI (adjust for your deployment)
        self.redirect_uri = os.getenv('OAUTH_REDIRECT_URI', 'http://localhost:8501')
        
        # Initialize providers
        self.providers = {}
        
        if self.google_client_id and self.google_client_secret:
            self.providers['google'] = GoogleOAuth(
                self.google_client_id, 
                self.google_client_secret, 
                self.redirect_uri
            )
        
        if self.github_client_id and self.github_client_secret:
            self.providers['github'] = GitHubOAuth(
                self.github_client_id, 
                self.github_client_secret, 
                self.redirect_uri
            )
    
    def get_provider(self, provider_name: str) -> Optional[OAuthProvider]:
        """Get OAuth provider by name."""
        return self.providers.get(provider_name)
    
    def is_provider_configured(self, provider_name: str) -> bool:
        """Check if OAuth provider is configured."""
        return provider_name in self.providers
    
    def generate_state(self) -> str:
        """Generate secure state parameter for OAuth."""
        return secrets.token_urlsafe(32)
    
    def get_authorization_url(self, provider_name: str) -> Optional[str]:
        """Get authorization URL for provider."""
        provider = self.get_provider(provider_name)
        if not provider:
            return None
        
        state = self.generate_state()
        # Store state in session for verification
        st.session_state[f'oauth_state_{provider_name}'] = state
        
        return provider.get_authorization_url(state)
    
    def handle_oauth_callback(self, provider_name: str, code: str, state: str) -> Optional[Dict[str, Any]]:
        """Handle OAuth callback and return user info."""
        provider = self.get_provider(provider_name)
        if not provider:
            return None
        
        # Verify state parameter
        stored_state = st.session_state.get(f'oauth_state_{provider_name}')
        if not stored_state or stored_state != state:
            print("OAuth state mismatch")
            return None
        
        # Exchange code for token
        token_data = provider.exchange_code_for_token(code, state)
        if not token_data:
            return None
        
        access_token = token_data.get('access_token')
        if not access_token:
            return None
        
        # Get user info
        user_info = provider.get_user_info(access_token)
        if user_info:
            # Clean up state
            if f'oauth_state_{provider_name}' in st.session_state:
                del st.session_state[f'oauth_state_{provider_name}']
        
        return user_info

# Global OAuth manager instance
oauth_manager = OAuthManager()
