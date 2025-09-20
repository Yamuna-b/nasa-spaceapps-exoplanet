"""
Accessibility utilities for the Exoplanet AI application.
Includes high-contrast modes, alt text helpers, and WCAG compliance features.
"""
import streamlit as st
from typing import Dict, Any, Optional
import json

class AccessibilityManager:
    """Manages accessibility features and preferences."""
    
    def __init__(self):
        pass  # Session state initialization is handled in main()
    
    def init_accessibility_state(self):
        """Initialize accessibility settings in session state."""
        # This is now handled in main() to avoid initialization issues
        pass
    
    def show_accessibility_controls(self):
        """Show accessibility control panel in sidebar."""
        with st.sidebar.expander("Accessibility", expanded=False):
            st.write("**Visual Accessibility**")
            
            high_contrast = st.checkbox(
                "High Contrast Mode",
                value=st.session_state.accessibility_settings['high_contrast'],
                help="Increases contrast for better visibility"
            )
            
            large_text = st.checkbox(
                "Large Text",
                value=st.session_state.accessibility_settings['large_text'],
                help="Increases text size throughout the application"
            )
            
            reduced_motion = st.checkbox(
                "Reduce Motion",
                value=st.session_state.accessibility_settings['reduced_motion'],
                help="Reduces animations and transitions"
            )
            
            st.write("**Screen Reader Support**")
            
            screen_reader = st.checkbox(
                "Screen Reader Mode",
                value=st.session_state.accessibility_settings['screen_reader_mode'],
                help="Optimizes interface for screen readers"
            )
            
            alt_text = st.checkbox(
                "Show Alt Text",
                value=st.session_state.accessibility_settings['alt_text_enabled'],
                help="Displays alternative text for images and charts"
            )
            
            # Update settings
            st.session_state.accessibility_settings.update({
                'high_contrast': high_contrast,
                'large_text': large_text,
                'reduced_motion': reduced_motion,
                'screen_reader_mode': screen_reader,
                'alt_text_enabled': alt_text
            })
            
            if st.button("Reset to Defaults", help="Reset all accessibility settings"):
                self.reset_accessibility_settings()
    
    def get_accessibility_css(self) -> str:
        """Generate CSS based on accessibility settings."""
        settings = st.session_state.accessibility_settings
        css = []
        
        # High contrast mode
        if settings['high_contrast']:
            css.append("""
            .high-contrast {
                background-color: #000000 !important;
                color: #ffffff !important;
            }
            .high-contrast .stButton > button {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 2px solid #ffffff !important;
            }
            .high-contrast .stSelectbox > div > div {
                background-color: #000000 !important;
                color: #ffffff !important;
                border: 2px solid #ffffff !important;
            }
            .high-contrast .stTextInput > div > div > input {
                background-color: #000000 !important;
                color: #ffffff !important;
                border: 2px solid #ffffff !important;
            }
            .high-contrast .metric-card {
                background-color: #333333 !important;
                color: #ffffff !important;
                border: 2px solid #ffffff !important;
            }
            """)
        
        # Large text mode
        if settings['large_text']:
            css.append("""
            .large-text {
                font-size: 1.2em !important;
            }
            .large-text h1 { font-size: 3em !important; }
            .large-text h2 { font-size: 2.5em !important; }
            .large-text h3 { font-size: 2em !important; }
            .large-text .stButton > button {
                font-size: 1.2em !important;
                padding: 0.75rem 1.5rem !important;
            }
            """)
        
        # Reduced motion
        if settings['reduced_motion']:
            css.append("""
            .reduced-motion * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
            """)
        
        # Screen reader optimizations
        if settings['screen_reader_mode']:
            css.append("""
            .sr-only {
                position: absolute;
                width: 1px;
                height: 1px;
                padding: 0;
                margin: -1px;
                overflow: hidden;
                clip: rect(0, 0, 0, 0);
                white-space: nowrap;
                border: 0;
            }
            .sr-only:focus {
                position: static;
                width: auto;
                height: auto;
                padding: inherit;
                margin: inherit;
                overflow: visible;
                clip: auto;
                white-space: normal;
            }
            """)
        
        return f"<style>{''.join(css)}</style>"
    
    def apply_accessibility_classes(self) -> str:
        """Get CSS classes to apply based on accessibility settings."""
        settings = st.session_state.accessibility_settings
        classes = []
        
        if settings['high_contrast']:
            classes.append('high-contrast')
        if settings['large_text']:
            classes.append('large-text')
        if settings['reduced_motion']:
            classes.append('reduced-motion')
        
        return ' '.join(classes)
    
    def reset_accessibility_settings(self):
        """Reset accessibility settings to defaults."""
        st.session_state.accessibility_settings = {
            'high_contrast': False,
            'large_text': False,
            'reduced_motion': False,
            'screen_reader_mode': False,
            'keyboard_navigation': True,
            'alt_text_enabled': True
        }
        st.success("Accessibility settings reset to defaults")
        st.rerun()

class AltTextManager:
    """Manages alternative text for images and visualizations."""
    
    @staticmethod
    def get_chart_alt_text(chart_type: str, data_summary: Dict[str, Any]) -> str:
        """Generate alt text for charts based on type and data."""
        alt_texts = {
            'bar': f"Bar chart showing {data_summary.get('title', 'data visualization')}. "
                   f"Contains {data_summary.get('data_points', 'multiple')} data points.",
            
            'line': f"Line chart displaying {data_summary.get('title', 'trend data')}. "
                    f"Shows progression over {data_summary.get('x_axis', 'time')}.",
            
            'scatter': f"Scatter plot of {data_summary.get('title', 'data points')}. "
                       f"X-axis: {data_summary.get('x_axis', 'variable 1')}, "
                       f"Y-axis: {data_summary.get('y_axis', 'variable 2')}.",
            
            'histogram': f"Histogram showing distribution of {data_summary.get('title', 'values')}. "
                         f"Contains {data_summary.get('bins', 'multiple')} bins.",
            
            'pie': f"Pie chart showing proportions of {data_summary.get('title', 'categories')}. "
                   f"Contains {data_summary.get('categories', 'multiple')} segments.",
            
            'heatmap': f"Heatmap visualization of {data_summary.get('title', 'correlation data')}. "
                       f"Shows relationships between variables using color intensity.",
            
            'confusion_matrix': f"Confusion matrix showing model performance. "
                                f"Displays true vs predicted classifications with "
                                f"{data_summary.get('accuracy', 'calculated')} accuracy."
        }
        
        return alt_texts.get(chart_type, f"Data visualization: {data_summary.get('title', 'Chart')}")
    
    @staticmethod
    def get_image_alt_text(image_type: str, context: str = "") -> str:
        """Generate alt text for images."""
        alt_texts = {
            'profile': f"Profile picture{' for ' + context if context else ''}",
            'logo': f"Exoplanet AI application logo",
            'icon': f"Icon representing {context}" if context else "Application icon",
            'chart': f"Chart visualization{' showing ' + context if context else ''}",
            'diagram': f"Diagram illustrating {context}" if context else "Technical diagram",
            'screenshot': f"Screenshot of {context}" if context else "Application screenshot"
        }
        
        return alt_texts.get(image_type, f"Image{' of ' + context if context else ''}")

class KeyboardNavigationManager:
    """Manages keyboard navigation and focus management."""
    
    @staticmethod
    def add_keyboard_shortcuts():
        """Add keyboard shortcuts for common actions."""
        shortcuts_js = """
        <script>
        document.addEventListener('keydown', function(e) {
            // Alt + H: Help
            if (e.altKey && e.key === 'h') {
                e.preventDefault();
                // Focus on help section or show help modal
                const helpButton = document.querySelector('[data-testid="help-button"]');
                if (helpButton) helpButton.click();
            }
            
            // Alt + M: Main content
            if (e.altKey && e.key === 'm') {
                e.preventDefault();
                const mainContent = document.querySelector('.main');
                if (mainContent) mainContent.focus();
            }
            
            // Alt + S: Sidebar
            if (e.altKey && e.key === 's') {
                e.preventDefault();
                const sidebar = document.querySelector('.sidebar');
                if (sidebar) sidebar.focus();
            }
            
            // Escape: Close modals/dropdowns
            if (e.key === 'Escape') {
                const modals = document.querySelectorAll('.modal, .dropdown-open');
                modals.forEach(modal => modal.style.display = 'none');
            }
        });
        </script>
        """
        st.components.v1.html(shortcuts_js, height=0)
    
    @staticmethod
    def add_skip_links():
        """Add skip navigation links for screen readers."""
        skip_links_html = """
        <div class="skip-links">
            <a href="#main-content" class="sr-only sr-only:focus">Skip to main content</a>
            <a href="#sidebar" class="sr-only sr-only:focus">Skip to navigation</a>
            <a href="#footer" class="sr-only sr-only:focus">Skip to footer</a>
        </div>
        """
        st.markdown(skip_links_html, unsafe_allow_html=True)

class ARIAManager:
    """Manages ARIA labels and attributes for better screen reader support."""
    
    @staticmethod
    def get_aria_label(element_type: str, context: str = "") -> str:
        """Generate appropriate ARIA labels."""
        aria_labels = {
            'button': f"Button{' to ' + context if context else ''}",
            'link': f"Link{' to ' + context if context else ''}",
            'input': f"Input field{' for ' + context if context else ''}",
            'select': f"Dropdown selection{' for ' + context if context else ''}",
            'chart': f"Interactive chart{' showing ' + context if context else ''}",
            'table': f"Data table{' containing ' + context if context else ''}",
            'form': f"Form{' for ' + context if context else ''}",
            'navigation': f"Navigation menu{' for ' + context if context else ''}",
            'main': "Main content area",
            'sidebar': "Sidebar navigation",
            'footer': "Page footer"
        }
        
        return aria_labels.get(element_type, context or "Interactive element")
    
    @staticmethod
    def add_aria_live_region():
        """Add ARIA live region for dynamic content updates."""
        live_region_html = """
        <div id="aria-live-region" aria-live="polite" aria-atomic="true" class="sr-only"></div>
        """
        st.markdown(live_region_html, unsafe_allow_html=True)
    
    @staticmethod
    def announce_to_screen_reader(message: str):
        """Announce message to screen readers via ARIA live region."""
        announce_js = f"""
        <script>
        const liveRegion = document.getElementById('aria-live-region');
        if (liveRegion) {{
            liveRegion.textContent = '{message}';
            setTimeout(() => liveRegion.textContent = '', 1000);
        }}
        </script>
        """
        st.components.v1.html(announce_js, height=0)

# Global accessibility instances
accessibility_manager = AccessibilityManager()
alt_text_manager = AltTextManager()
keyboard_nav_manager = KeyboardNavigationManager()
aria_manager = ARIAManager()
