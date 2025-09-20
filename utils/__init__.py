"""
Utility modules for Exoplanet AI application.
"""

from .accessibility import accessibility_manager, alt_text_manager, keyboard_nav_manager, aria_manager
from .error_handler import error_handler, handle_errors
from .help_system import help_system
from .contact_footer import contact_form, footer_system
from .api_docs import api_docs

__all__ = [
    'accessibility_manager',
    'alt_text_manager', 
    'keyboard_nav_manager',
    'aria_manager',
    'error_handler',
    'handle_errors',
    'help_system',
    'contact_form',
    'footer_system',
    'api_docs'
]
