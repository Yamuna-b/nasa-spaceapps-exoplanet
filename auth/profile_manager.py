"""
Profile management interface for user settings and preferences.
"""
import streamlit as st
from typing import Dict, Any, Optional
import json
from datetime import datetime

from .database import db
from .auth_utils import password_validator, password_hasher

class ProfileManager:
    """Profile management for authenticated users."""
    
    def show_profile_page(self, user: Dict[str, Any]):
        """Show user profile management page."""
        st.header("👤 Profile Management")
        
        # Profile tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Profile Info", "Security", "Preferences", "Account"])
        
        with tab1:
            self.show_profile_info_tab(user)
        
        with tab2:
            self.show_security_tab(user)
        
        with tab3:
            self.show_preferences_tab(user)
        
        with tab4:
            self.show_account_tab(user)
    
    def show_profile_info_tab(self, user: Dict[str, Any]):
        """Show profile information tab."""
        st.subheader("📝 Profile Information")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Profile picture section
            st.write("**Profile Picture**")
            
            current_picture = user.get('profile_picture_url', '')
            if current_picture:
                st.image(current_picture, width=150)
            else:
                st.info("No profile picture set")
            
            # Profile picture upload/URL
            picture_option = st.radio("Profile Picture", ["Upload File", "Use URL", "Remove"])
            
            new_picture_url = None
            if picture_option == "Upload File":
                uploaded_file = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg'])
                if uploaded_file:
                    # In a real app, you'd upload to cloud storage
                    st.info("File upload would be implemented with cloud storage (AWS S3, etc.)")
            elif picture_option == "Use URL":
                new_picture_url = st.text_input("Image URL", value=current_picture)
            elif picture_option == "Remove":
                new_picture_url = ""
        
        with col2:
            # Profile form
            with st.form("profile_form"):
                st.write("**Basic Information**")
                
                name = st.text_input("Full Name", value=user.get('name', ''))
                email_display = st.text_input("Email", value=user.get('email', ''), disabled=True)
                st.caption("Email cannot be changed. Contact support if needed.")
                
                bio = st.text_area(
                    "Bio", 
                    value=user.get('bio', ''), 
                    max_chars=500,
                    help="Tell others about yourself (max 500 characters)"
                )
                
                institution = st.text_input(
                    "Institution/Organization", 
                    value=user.get('institution', ''),
                    help="University, company, or research institution"
                )
                
                research_interests = st.text_area(
                    "Research Interests", 
                    value=user.get('research_interests', ''),
                    help="Your areas of interest in astronomy, exoplanets, or related fields"
                )
                
                # Additional fields
                st.write("**Additional Information**")
                
                # Parse preferences for additional fields
                preferences = json.loads(user.get('preferences', '{}'))
                
                location = st.text_input(
                    "Location", 
                    value=preferences.get('location', ''),
                    help="City, Country"
                )
                
                website = st.text_input(
                    "Website/Portfolio", 
                    value=preferences.get('website', ''),
                    help="Your personal website or portfolio URL"
                )
                
                orcid = st.text_input(
                    "ORCID ID", 
                    value=preferences.get('orcid', ''),
                    help="Your ORCID identifier (for researchers)"
                )
                
                # Privacy settings
                st.write("**Privacy Settings**")
                
                profile_public = st.checkbox(
                    "Make profile public", 
                    value=preferences.get('profile_public', True),
                    help="Allow other users to see your profile"
                )
                
                show_email = st.checkbox(
                    "Show email to other users", 
                    value=preferences.get('show_email', False),
                    help="Display your email address on your public profile"
                )
                
                allow_collaboration = st.checkbox(
                    "Allow collaboration invites", 
                    value=preferences.get('allow_collaboration', True),
                    help="Allow other users to invite you to collaborate"
                )
                
                update_btn = st.form_submit_button("Update Profile", use_container_width=True)
                
                if update_btn:
                    self.update_profile(
                        user['user_id'], name, bio, institution, research_interests,
                        new_picture_url, location, website, orcid, profile_public,
                        show_email, allow_collaboration
                    )
    
    def show_security_tab(self, user: Dict[str, Any]):
        """Show security settings tab."""
        st.subheader("🔒 Security Settings")
        
        # Password change section
        st.write("**Change Password**")
        
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
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
            
            change_password_btn = st.form_submit_button("Change Password")
            
            if change_password_btn:
                self.change_password(user, current_password, new_password, confirm_password)
        
        st.divider()
        
        # Account activity
        st.write("**Account Activity**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Account Created", user.get('created_at', 'Unknown'))
            st.metric("Email Verified", "Yes" if user.get('is_verified') else "No")
        
        with col2:
            last_login = user.get('last_login', 'Never')
            st.metric("Last Login", last_login)
            st.metric("Account Status", "Active" if user.get('is_active') else "Inactive")
        
        # Session management
        st.write("**Active Sessions**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Sign Out All Devices", type="secondary"):
                self.sign_out_all_devices(user['user_id'])
        
        with col2:
            st.info("Current session will remain active")
        
        st.divider()
        
        # Two-factor authentication (placeholder)
        st.write("**Two-Factor Authentication**")
        st.info("🚧 Two-factor authentication coming soon!")
        
        enable_2fa = st.checkbox("Enable 2FA (when available)", disabled=True)
    
    def show_preferences_tab(self, user: Dict[str, Any]):
        """Show user preferences tab."""
        st.subheader("⚙️ Application Preferences")
        
        # Parse current preferences
        preferences = json.loads(user.get('preferences', '{}'))
        
        with st.form("preferences_form"):
            st.write("**Email Notifications**")
            
            email_notifications = st.checkbox(
                "Enable email notifications", 
                value=preferences.get('email_notifications', True)
            )
            
            if email_notifications:
                analysis_complete = st.checkbox(
                    "Analysis completion notifications", 
                    value=preferences.get('notify_analysis_complete', True)
                )
                
                weekly_summary = st.checkbox(
                    "Weekly summary emails", 
                    value=preferences.get('notify_weekly_summary', False)
                )
                
                collaboration_invites = st.checkbox(
                    "Collaboration invite notifications", 
                    value=preferences.get('notify_collaboration', True)
                )
                
                system_announcements = st.checkbox(
                    "System announcements", 
                    value=preferences.get('notify_system', True)
                )
            else:
                analysis_complete = weekly_summary = collaboration_invites = system_announcements = False
            
            st.write("**Application Settings**")
            
            default_dataset = st.selectbox(
                "Default Dataset", 
                ["kepler", "tess", "k2"],
                index=["kepler", "tess", "k2"].index(preferences.get('default_dataset', 'kepler'))
            )
            
            default_model = st.selectbox(
                "Default Model", 
                ["rf", "logreg", "svm"],
                index=["rf", "logreg", "svm"].index(preferences.get('default_model', 'rf'))
            )
            
            visualization_style = st.selectbox(
                "Preferred Visualization Style", 
                ["plotly", "matplotlib", "seaborn"],
                index=["plotly", "matplotlib", "seaborn"].index(preferences.get('viz_style', 'plotly'))
            )
            
            export_format = st.selectbox(
                "Default Export Format", 
                ["csv", "json", "excel"],
                index=["csv", "json", "excel"].index(preferences.get('export_format', 'csv'))
            )
            
            st.write("**Interface Settings**")
            
            theme = st.selectbox(
                "Theme", 
                ["auto", "light", "dark"],
                index=["auto", "light", "dark"].index(preferences.get('theme', 'auto'))
            )
            
            sidebar_collapsed = st.checkbox(
                "Collapse sidebar by default", 
                value=preferences.get('sidebar_collapsed', False)
            )
            
            show_tooltips = st.checkbox(
                "Show helpful tooltips", 
                value=preferences.get('show_tooltips', True)
            )
            
            save_preferences_btn = st.form_submit_button("Save Preferences", use_container_width=True)
            
            if save_preferences_btn:
                new_preferences = {
                    'email_notifications': email_notifications,
                    'notify_analysis_complete': analysis_complete,
                    'notify_weekly_summary': weekly_summary,
                    'notify_collaboration': collaboration_invites,
                    'notify_system': system_announcements,
                    'default_dataset': default_dataset,
                    'default_model': default_model,
                    'viz_style': visualization_style,
                    'export_format': export_format,
                    'theme': theme,
                    'sidebar_collapsed': sidebar_collapsed,
                    'show_tooltips': show_tooltips,
                    # Preserve existing preferences
                    **{k: v for k, v in preferences.items() if k not in new_preferences}
                }
                
                self.update_preferences(user['user_id'], new_preferences)
    
    def show_account_tab(self, user: Dict[str, Any]):
        """Show account management tab."""
        st.subheader("🏠 Account Management")
        
        # Account information
        st.write("**Account Information**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**User ID:** {user['user_id']}")
            st.info(f"**Email:** {user['email']}")
            st.info(f"**Member Since:** {user.get('created_at', 'Unknown')}")
        
        with col2:
            st.info(f"**Account Type:** Standard User")
            st.info(f"**Status:** {'Active' if user.get('is_active') else 'Inactive'}")
            st.info(f"**Email Verified:** {'Yes' if user.get('is_verified') else 'No'}")
        
        st.divider()
        
        # Connected accounts
        st.write("**Connected Accounts**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            google_connected = bool(user.get('google_id'))
            if google_connected:
                st.success("✅ Google Account Connected")
                if st.button("Disconnect Google", type="secondary"):
                    self.disconnect_oauth_account(user['user_id'], 'google')
            else:
                st.info("❌ Google Account Not Connected")
                if st.button("Connect Google", type="secondary"):
                    st.info("OAuth connection would be implemented here")
        
        with col2:
            github_connected = bool(user.get('github_id'))
            if github_connected:
                st.success("✅ GitHub Account Connected")
                if st.button("Disconnect GitHub", type="secondary"):
                    self.disconnect_oauth_account(user['user_id'], 'github')
            else:
                st.info("❌ GitHub Account Not Connected")
                if st.button("Connect GitHub", type="secondary"):
                    st.info("OAuth connection would be implemented here")
        
        st.divider()
        
        # Data export
        st.write("**Data Export**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download My Data", type="secondary"):
                self.export_user_data(user)
        
        with col2:
            st.caption("Export all your profile data and analysis history")
        
        st.divider()
        
        # Danger zone
        st.write("**⚠️ Danger Zone**")
        
        with st.expander("Account Deactivation", expanded=False):
            st.warning("Deactivating your account will:")
            st.write("- Hide your profile from other users")
            st.write("- Prevent you from signing in")
            st.write("- Preserve your data for potential reactivation")
            
            if st.button("Deactivate Account", type="secondary"):
                self.deactivate_account(user['user_id'])
        
        with st.expander("Delete Account", expanded=False):
            st.error("⚠️ **This action cannot be undone!**")
            st.write("Deleting your account will:")
            st.write("- Permanently remove all your data")
            st.write("- Delete your profile and analysis history")
            st.write("- Cannot be reversed")
            
            delete_confirmation = st.text_input(
                "Type 'DELETE' to confirm account deletion:",
                placeholder="DELETE"
            )
            
            if delete_confirmation == "DELETE":
                if st.button("🗑️ Delete Account Permanently", type="secondary"):
                    self.delete_account(user['user_id'])
    
    def update_profile(self, user_id: str, name: str, bio: str, institution: str, 
                      research_interests: str, picture_url: str, location: str, 
                      website: str, orcid: str, profile_public: bool, 
                      show_email: bool, allow_collaboration: bool):
        """Update user profile information."""
        try:
            # Update basic profile fields
            success = db.update_user_profile(
                user_id,
                name=name,
                bio=bio,
                institution=institution,
                research_interests=research_interests,
                profile_picture_url=picture_url
            )
            
            if success:
                # Update preferences with additional fields
                user = db.get_user_by_id(user_id)
                current_preferences = json.loads(user.get('preferences', '{}'))
                
                current_preferences.update({
                    'location': location,
                    'website': website,
                    'orcid': orcid,
                    'profile_public': profile_public,
                    'show_email': show_email,
                    'allow_collaboration': allow_collaboration
                })
                
                self.update_preferences(user_id, current_preferences)
                
                # Update session state
                updated_user = db.get_user_by_id(user_id)
                st.session_state.user = updated_user
                
                st.success("Profile updated successfully!")
            else:
                st.error("Failed to update profile. Please try again.")
                
        except Exception as e:
            st.error(f"Error updating profile: {e}")
    
    def change_password(self, user: Dict[str, Any], current_password: str, 
                       new_password: str, confirm_password: str):
        """Change user password."""
        if not all([current_password, new_password, confirm_password]):
            st.error("Please fill in all password fields")
            return
        
        # Verify current password
        if not password_hasher.verify_password(current_password, user['password_hash']):
            st.error("Current password is incorrect")
            return
        
        # Validate new password
        password_valid, password_msg = password_validator.validate_password(new_password)
        if not password_valid:
            st.error(password_msg)
            return
        
        # Check password confirmation
        if new_password != confirm_password:
            st.error("New passwords do not match")
            return
        
        # Check if new password is different from current
        if password_hasher.verify_password(new_password, user['password_hash']):
            st.error("New password must be different from current password")
            return
        
        try:
            # Update password
            new_password_hash = password_hasher.hash_password(new_password)
            
            # Update in database (we need to add this method to DatabaseManager)
            with db.db_path as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE users SET password_hash = ? WHERE user_id = ?",
                    (new_password_hash, user['user_id'])
                )
                conn.commit()
            
            st.success("Password changed successfully!")
            
        except Exception as e:
            st.error(f"Error changing password: {e}")
    
    def update_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences."""
        try:
            preferences_json = json.dumps(preferences)
            
            with db.db_path as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE users SET preferences = ? WHERE user_id = ?",
                    (preferences_json, user_id)
                )
                conn.commit()
            
            # Update session state
            updated_user = db.get_user_by_id(user_id)
            st.session_state.user = updated_user
            
            st.success("Preferences saved successfully!")
            
        except Exception as e:
            st.error(f"Error saving preferences: {e}")
    
    def sign_out_all_devices(self, user_id: str):
        """Sign out user from all devices."""
        try:
            db.delete_all_user_sessions(user_id)
            st.success("Signed out from all devices successfully!")
            st.info("You will need to sign in again on other devices.")
            
        except Exception as e:
            st.error(f"Error signing out from all devices: {e}")
    
    def disconnect_oauth_account(self, user_id: str, provider: str):
        """Disconnect OAuth account."""
        try:
            field = f"{provider}_id"
            
            with db.db_path as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"UPDATE users SET {field} = NULL WHERE user_id = ?",
                    (user_id,)
                )
                conn.commit()
            
            # Update session state
            updated_user = db.get_user_by_id(user_id)
            st.session_state.user = updated_user
            
            st.success(f"{provider.title()} account disconnected successfully!")
            
        except Exception as e:
            st.error(f"Error disconnecting {provider} account: {e}")
    
    def export_user_data(self, user: Dict[str, Any]):
        """Export user data."""
        try:
            # Create user data export
            export_data = {
                'profile': {
                    'name': user.get('name'),
                    'email': user.get('email'),
                    'bio': user.get('bio'),
                    'institution': user.get('institution'),
                    'research_interests': user.get('research_interests'),
                    'created_at': user.get('created_at'),
                    'last_login': user.get('last_login')
                },
                'preferences': json.loads(user.get('preferences', '{}')),
                'export_date': datetime.now().isoformat()
            }
            
            # Convert to JSON
            export_json = json.dumps(export_data, indent=2)
            
            # Provide download
            st.download_button(
                label="📥 Download Data (JSON)",
                data=export_json,
                file_name=f"exoplanet_ai_data_{user['user_id']}.json",
                mime="application/json"
            )
            
            st.success("Data export ready for download!")
            
        except Exception as e:
            st.error(f"Error exporting data: {e}")
    
    def deactivate_account(self, user_id: str):
        """Deactivate user account."""
        try:
            with db.db_path as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE users SET is_active = FALSE WHERE user_id = ?",
                    (user_id,)
                )
                conn.commit()
            
            # Sign out all sessions
            db.delete_all_user_sessions(user_id)
            
            st.success("Account deactivated successfully!")
            st.info("Contact support to reactivate your account.")
            
            # Force logout
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
            
        except Exception as e:
            st.error(f"Error deactivating account: {e}")
    
    def delete_account(self, user_id: str):
        """Permanently delete user account."""
        try:
            with db.db_path as conn:
                cursor = conn.cursor()
                
                # Delete user sessions
                cursor.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
                
                # Delete login attempts
                user = db.get_user_by_id(user_id)
                if user:
                    cursor.execute("DELETE FROM login_attempts WHERE email = ?", (user['email'],))
                
                # Delete user
                cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                
                conn.commit()
            
            st.success("Account deleted permanently!")
            st.info("Thank you for using Exoplanet AI. Your data has been removed.")
            
            # Force logout
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
            
        except Exception as e:
            st.error(f"Error deleting account: {e}")

# Global profile manager instance
profile_manager = ProfileManager()
