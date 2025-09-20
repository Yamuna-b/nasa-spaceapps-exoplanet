"""
Error handling, logging, and issue reporting system.
"""
import streamlit as st
import logging
import traceback
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class ErrorLogger:
    """Centralized error logging system."""
    
    def __init__(self, log_file: str = "logs/exoplanet_ai.log"):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('ExoplanetAI')
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None, user_id: str = None):
        """Log an error with context information."""
        error_id = str(uuid.uuid4())
        
        error_info = {
            'error_id': error_id,
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'user_id': user_id,
            'context': context or {},
            'session_state': self._get_safe_session_state()
        }
        
        self.logger.error(f"Error {error_id}: {json.dumps(error_info, indent=2)}")
        
        return error_id
    
    def log_info(self, message: str, context: Dict[str, Any] = None, user_id: str = None):
        """Log an info message."""
        info = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'user_id': user_id,
            'context': context or {}
        }
        
        self.logger.info(json.dumps(info))
    
    def log_warning(self, message: str, context: Dict[str, Any] = None, user_id: str = None):
        """Log a warning message."""
        warning = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'user_id': user_id,
            'context': context or {}
        }
        
        self.logger.warning(json.dumps(warning))
    
    def _get_safe_session_state(self) -> Dict[str, Any]:
        """Get session state without sensitive information."""
        safe_state = {}
        
        for key, value in st.session_state.items():
            # Skip sensitive information
            if any(sensitive in key.lower() for sensitive in ['password', 'token', 'secret', 'key']):
                safe_state[key] = '[REDACTED]'
            elif isinstance(value, (str, int, float, bool, list, dict)):
                try:
                    json.dumps(value)  # Test if serializable
                    safe_state[key] = value
                except (TypeError, ValueError):
                    safe_state[key] = str(type(value))
            else:
                safe_state[key] = str(type(value))
        
        return safe_state

class IssueReporter:
    """Issue reporting system for users to report bugs and feedback."""
    
    def __init__(self):
        self.error_logger = ErrorLogger()
    
    def show_report_issue_button(self):
        """Show the report issue button in the sidebar."""
        if st.sidebar.button("Report Issue", help="Report a bug or provide feedback"):
            st.session_state.show_issue_form = True
    
    def show_issue_form(self):
        """Show the issue reporting form."""
        if not st.session_state.get('show_issue_form', False):
            return
        
        st.markdown("### Report an Issue")
        
        with st.form("issue_report_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                issue_type = st.selectbox(
                    "Issue Type",
                    ["Bug Report", "Feature Request", "Performance Issue", "UI/UX Feedback", "Other"],
                    help="Select the type of issue you're reporting"
                )
                
                severity = st.selectbox(
                    "Severity",
                    ["Low", "Medium", "High", "Critical"],
                    index=1,
                    help="How severe is this issue?"
                )
            
            with col2:
                page_location = st.text_input(
                    "Page/Section",
                    value=st.session_state.get('current_page', ''),
                    help="Which page or section were you using?"
                )
                
                browser_info = st.text_input(
                    "Browser (Optional)",
                    placeholder="e.g., Chrome 120, Firefox 115",
                    help="Your browser and version (optional)"
                )
            
            title = st.text_input(
                "Issue Title",
                placeholder="Brief description of the issue",
                help="A short, descriptive title for your issue"
            )
            
            description = st.text_area(
                "Detailed Description",
                placeholder="Please describe the issue in detail. Include steps to reproduce if it's a bug.",
                height=150,
                help="The more details you provide, the better we can help you"
            )
            
            steps_to_reproduce = st.text_area(
                "Steps to Reproduce (for bugs)",
                placeholder="1. Go to...\n2. Click on...\n3. See error...",
                height=100,
                help="Step-by-step instructions to reproduce the issue"
            )
            
            expected_behavior = st.text_input(
                "Expected Behavior (for bugs)",
                placeholder="What did you expect to happen?",
                help="Describe what you expected to happen instead"
            )
            
            contact_email = st.text_input(
                "Your Email (Optional)",
                placeholder="your.email@example.com",
                help="We'll only use this to follow up on your report if needed"
            )
            
            include_logs = st.checkbox(
                "Include technical information",
                value=True,
                help="Include session information to help us debug (no personal data)"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                submit_btn = st.form_submit_button("Submit Report", type="primary")
            
            with col2:
                cancel_btn = st.form_submit_button("Cancel")
            
            with col3:
                st.caption("Reports help improve the app for everyone!")
            
            if submit_btn:
                if title and description:
                    issue_id = self.submit_issue_report({
                        'type': issue_type,
                        'severity': severity,
                        'title': title,
                        'description': description,
                        'steps_to_reproduce': steps_to_reproduce,
                        'expected_behavior': expected_behavior,
                        'page_location': page_location,
                        'browser_info': browser_info,
                        'contact_email': contact_email,
                        'include_logs': include_logs
                    })
                    
                    if issue_id:
                        st.success(f"✅ Issue reported successfully! Reference ID: {issue_id}")
                        st.info("Thank you for helping us improve Exoplanet AI!")
                        st.session_state.show_issue_form = False
                        st.rerun()
                    else:
                        st.error("Failed to submit issue report. Please try again.")
                else:
                    st.error("Please fill in the title and description fields.")
            
            if cancel_btn:
                st.session_state.show_issue_form = False
                st.rerun()
    
    def submit_issue_report(self, issue_data: Dict[str, Any]) -> Optional[str]:
        """Submit an issue report."""
        try:
            issue_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Get current user if authenticated
            user_id = None
            user_email = None
            if hasattr(st.session_state, 'user') and st.session_state.user:
                user_id = st.session_state.user.get('user_id')
                user_email = st.session_state.user.get('email')
            
            # Prepare issue report
            report = {
                'issue_id': issue_id,
                'timestamp': timestamp,
                'user_id': user_id,
                'user_email': user_email,
                'contact_email': issue_data.get('contact_email'),
                'type': issue_data['type'],
                'severity': issue_data['severity'],
                'title': issue_data['title'],
                'description': issue_data['description'],
                'steps_to_reproduce': issue_data.get('steps_to_reproduce'),
                'expected_behavior': issue_data.get('expected_behavior'),
                'page_location': issue_data.get('page_location'),
                'browser_info': issue_data.get('browser_info'),
                'technical_info': self._get_technical_info() if issue_data.get('include_logs') else None
            }
            
            # Save to file
            self._save_issue_report(report)
            
            # Send email notification (if configured)
            self._send_issue_notification(report)
            
            # Log the issue
            self.error_logger.log_info(
                f"Issue reported: {issue_data['title']}",
                context={'issue_id': issue_id, 'type': issue_data['type']},
                user_id=user_id
            )
            
            return issue_id
            
        except Exception as e:
            self.error_logger.log_error(e, context={'action': 'submit_issue_report'})
            return None
    
    def _save_issue_report(self, report: Dict[str, Any]):
        """Save issue report to file."""
        reports_dir = "logs/issue_reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        filename = f"{reports_dir}/issue_{report['issue_id']}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _send_issue_notification(self, report: Dict[str, Any]):
        """Send email notification about new issue (if configured)."""
        try:
            # This would be configured with actual email settings in production
            admin_email = os.getenv('ADMIN_EMAIL', 'admin@exoplanet-ai.com')
            
            subject = f"[Exoplanet AI] New {report['type']}: {report['title']}"
            
            body = f"""
            New issue reported in Exoplanet AI:
            
            Issue ID: {report['issue_id']}
            Type: {report['type']}
            Severity: {report['severity']}
            Title: {report['title']}
            
            Description:
            {report['description']}
            
            Reporter: {report.get('contact_email', 'Anonymous')}
            User ID: {report.get('user_id', 'Not authenticated')}
            Page: {report.get('page_location', 'Unknown')}
            Browser: {report.get('browser_info', 'Unknown')}
            
            Timestamp: {report['timestamp']}
            """
            
            # In development, just print the email
            print(f"\n--- ISSUE REPORT EMAIL ---")
            print(f"To: {admin_email}")
            print(f"Subject: {subject}")
            print(f"Body: {body}")
            print("--- END EMAIL ---\n")
            
        except Exception as e:
            self.error_logger.log_error(e, context={'action': 'send_issue_notification'})
    
    def _get_technical_info(self) -> Dict[str, Any]:
        """Get technical information for debugging."""
        return {
            'timestamp': datetime.now().isoformat(),
            'session_state_keys': list(st.session_state.keys()),
            'url_params': dict(st.query_params) if hasattr(st, 'query_params') else {},
            'user_agent': 'Not available in Streamlit',  # Would need JavaScript to get this
            'screen_resolution': 'Not available in Streamlit'
        }

class ErrorHandler:
    """Main error handler with user-friendly error display."""
    
    def __init__(self):
        self.error_logger = ErrorLogger()
        self.issue_reporter = IssueReporter()
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None, 
                    show_to_user: bool = True, user_friendly_message: str = None):
        """Handle an error with logging and user notification."""
        # Get current user if available
        user_id = None
        if hasattr(st.session_state, 'user') and st.session_state.user:
            user_id = st.session_state.user.get('user_id')
        
        # Log the error
        error_id = self.error_logger.log_error(error, context, user_id)
        
        if show_to_user:
            self.show_error_to_user(error, error_id, user_friendly_message)
        
        return error_id
    
    def show_error_to_user(self, error: Exception, error_id: str, 
                          user_friendly_message: str = None):
        """Show user-friendly error message."""
        if user_friendly_message:
            st.error(user_friendly_message)
        else:
            st.error("An unexpected error occurred. Please try again.")
        
        with st.expander("Technical Details", expanded=False):
            st.code(f"Error ID: {error_id}")
            st.code(f"Error Type: {type(error).__name__}")
            st.code(f"Error Message: {str(error)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🐛 Report This Error", key=f"report_{error_id}"):
                    st.session_state.show_issue_form = True
                    st.session_state.prefill_error_id = error_id
            
            with col2:
                st.caption(f"Reference this ID when reporting: {error_id}")
    
    def safe_execute(self, func, *args, **kwargs):
        """Safely execute a function with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, context={'function': func.__name__})
            return None
    
    def show_error_boundary(self, component_name: str):
        """Show error boundary for a component."""
        try:
            yield
        except Exception as e:
            self.handle_error(
                e, 
                context={'component': component_name},
                user_friendly_message=f"Error in {component_name}. Please refresh the page or try again."
            )

# Global error handler instance
error_handler = ErrorHandler()

# Decorator for error handling
def handle_errors(user_friendly_message: str = None):
    """Decorator to handle errors in functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(
                    e, 
                    context={'function': func.__name__},
                    user_friendly_message=user_friendly_message
                )
                return None
        return wrapper
    return decorator
