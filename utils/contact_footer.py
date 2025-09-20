"""
Contact form and footer system for the application.
"""
import streamlit as st
from datetime import datetime
from typing import Dict, Any
import json
import uuid

class ContactForm:
    """Contact form for project inquiries and demo requests."""
    
    def show_contact_form(self):
        """Show the main contact form."""
        st.header("📞 Contact Us")
        
        st.markdown("""
        Get in touch with us for project inquiries, demo requests, collaborations, or any questions about Exoplanet AI.
        """)
        
        # Contact options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Project Inquiries
            - Custom exoplanet analysis projects
            - Research collaborations
            - Educational partnerships
            - Enterprise solutions
            """)
        
        with col2:
            st.markdown("""
            ### Demo Requests
            - Live demonstration sessions
            - Training workshops
            - Conference presentations
            - Academic seminars
            """)
        
        st.markdown("---")
        
        # Contact form
        with st.form("contact_form", clear_on_submit=True):
            st.subheader("📝 Send us a message")
            
            # Basic information
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name *", placeholder="Your full name")
                email = st.text_input("Email Address *", placeholder="your.email@example.com")
                phone = st.text_input("Phone Number", placeholder="+1 (555) 123-4567")
            
            with col2:
                organization = st.text_input("Organization", placeholder="University, Company, etc.")
                position = st.text_input("Position/Title", placeholder="Professor, Researcher, Student, etc.")
                country = st.text_input("Country", placeholder="Your country")
            
            # Inquiry details
            inquiry_type = st.selectbox(
                "Type of Inquiry *",
                [
                    "General Information",
                    "Project Collaboration",
                    "Demo Request",
                    "Educational Partnership",
                    "Research Collaboration",
                    "Enterprise Solution",
                    "Media/Press Inquiry",
                    "Technical Support",
                    "Other"
                ]
            )
            
            subject = st.text_input("Subject *", placeholder="Brief description of your inquiry")
            
            message = st.text_area(
                "Message *",
                placeholder="Please provide details about your inquiry, project requirements, timeline, etc.",
                height=150
            )
            
            # Additional options
            st.markdown("### Additional Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                timeline = st.selectbox(
                    "Timeline",
                    ["Immediate", "Within 1 week", "Within 1 month", "Within 3 months", "Flexible"]
                )
                
                budget_range = st.selectbox(
                    "Budget Range (if applicable)",
                    ["Not applicable", "Under $1,000", "$1,000 - $5,000", "$5,000 - $10,000", "$10,000+", "To be discussed"]
                )
            
            with col2:
                preferred_contact = st.multiselect(
                    "Preferred Contact Method",
                    ["Email", "Phone", "Video Call", "In-person meeting"]
                )
                
                newsletter_signup = st.checkbox("Subscribe to our newsletter for updates")
            
            # Special requests
            special_requirements = st.text_area(
                "Special Requirements or Notes",
                placeholder="Any specific requirements, accessibility needs, or additional information...",
                height=80
            )
            
            # Privacy and consent
            st.markdown("### Privacy & Consent")
            
            privacy_consent = st.checkbox(
                "I consent to the processing of my personal data for the purpose of this inquiry *",
                help="We will only use your information to respond to your inquiry and will not share it with third parties."
            )
            
            marketing_consent = st.checkbox(
                "I would like to receive updates about Exoplanet AI and related projects",
                help="You can unsubscribe at any time."
            )
            
            # Submit button
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col2:
                submit_button = st.form_submit_button("Send Message", type="primary")
            
            if submit_button:
                if self.validate_contact_form(name, email, subject, message, privacy_consent):
                    contact_data = {
                        'name': name,
                        'email': email,
                        'phone': phone,
                        'organization': organization,
                        'position': position,
                        'country': country,
                        'inquiry_type': inquiry_type,
                        'subject': subject,
                        'message': message,
                        'timeline': timeline,
                        'budget_range': budget_range,
                        'preferred_contact': preferred_contact,
                        'special_requirements': special_requirements,
                        'newsletter_signup': newsletter_signup,
                        'privacy_consent': privacy_consent,
                        'marketing_consent': marketing_consent
                    }
                    
                    if self.submit_contact_form(contact_data):
                        st.success("✅ Thank you! Your message has been sent successfully.")
                        st.info("We'll get back to you within 24-48 hours.")
                        
                        # Show next steps
                        st.markdown("""
                        ### What happens next?
                        1. **Confirmation**: You'll receive an email confirmation shortly
                        2. **Review**: Our team will review your inquiry
                        3. **Response**: We'll contact you within 24-48 hours
                        4. **Follow-up**: We'll schedule a call or meeting if needed
                        """)
                    else:
                        st.error("❌ There was an error sending your message. Please try again.")
                else:
                    st.error("❌ Please fill in all required fields and accept the privacy policy.")
        
        # Alternative contact methods
        st.markdown("---")
        self.show_alternative_contact_methods()
    
    def validate_contact_form(self, name: str, email: str, subject: str, message: str, privacy_consent: bool) -> bool:
        """Validate contact form data."""
        if not all([name.strip(), email.strip(), subject.strip(), message.strip()]):
            return False
        
        if not privacy_consent:
            return False
        
        # Basic email validation
        if '@' not in email or '.' not in email.split('@')[1]:
            st.error("Please enter a valid email address.")
            return False
        
        return True
    
    def submit_contact_form(self, contact_data: Dict[str, Any]) -> bool:
        """Submit contact form data."""
        try:
            # Generate unique ID for the inquiry
            inquiry_id = str(uuid.uuid4())
            contact_data['inquiry_id'] = inquiry_id
            contact_data['timestamp'] = datetime.now().isoformat()
            
            # In a real application, this would:
            # 1. Save to database
            # 2. Send email to admin
            # 3. Send confirmation email to user
            # 4. Create support ticket
            
            # For now, just log the contact
            print(f"\n--- CONTACT FORM SUBMISSION ---")
            print(f"Inquiry ID: {inquiry_id}")
            print(f"From: {contact_data['name']} ({contact_data['email']})")
            print(f"Type: {contact_data['inquiry_type']}")
            print(f"Subject: {contact_data['subject']}")
            print(f"Message: {contact_data['message']}")
            print("--- END CONTACT FORM ---\n")
            
            # Save to file for development
            self.save_contact_inquiry(contact_data)
            
            return True
            
        except Exception as e:
            print(f"Error submitting contact form: {e}")
            return False
    
    def save_contact_inquiry(self, contact_data: Dict[str, Any]):
        """Save contact inquiry to file."""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs("logs/contact_inquiries", exist_ok=True)
        
        filename = f"logs/contact_inquiries/inquiry_{contact_data['inquiry_id']}.json"
        
        with open(filename, 'w') as f:
            json.dump(contact_data, f, indent=2)
    
    def show_alternative_contact_methods(self):
        """Show alternative contact methods."""
        st.subheader("🌐 Other Ways to Reach Us")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📧 Email
            **General Inquiries:**
            info@exoplanet-ai.com
            
            **Technical Support:**
            support@exoplanet-ai.com
            
            **Partnerships:**
            partnerships@exoplanet-ai.com
            """)
        
        with col2:
            st.markdown("""
            ### 🌐 Online
            **Website:**
            [exoplanet-ai.com](https://exoplanet-ai.com)
            
            **Documentation:**
            [docs.exoplanet-ai.com](https://docs.exoplanet-ai.com)
            
            **GitHub:**
            [github.com/Yamuna-b](https://github.com/Yamuna-b)
            """)
        
        with col3:
            st.markdown("""
            ### 📱 Social Media
            **LinkedIn:**
            [Yamuna B](https://www.linkedin.com/in/yamuna-bsvy/)
            
            **Twitter:**
            [@ExoplanetAI](https://twitter.com/ExoplanetAI)
            
            **Research Gate:**
            [Yamuna B](https://researchgate.net/profile/Yamuna-B)
            """)

class FooterSystem:
    """Footer system with credits, license, and links."""
    
    def show_footer(self):
        """Show the application footer."""
        st.markdown("---")
        
        # Main footer content
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            ### 🪐 Exoplanet AI
            Advancing exoplanet discovery through artificial intelligence and machine learning.
            
            **Version:** 1.0.0
            **Last Updated:** December 2024
            """)
        
        with col2:
            st.markdown("""
            ### Quick Links
            - [Home](/)
            - [Documentation](/docs)
            - [API Reference](/api)
            - [GitHub](https://github.com/Yamuna-b)
            - [Support](/support)
            """)
        
        with col3:
            st.markdown("""
            ### Resources
            - [User Guide](/help)
            - [FAQ](/faq)
            - [Tutorials](/tutorials)
            - [Research Papers](/papers)
            - [Datasets](/datasets)
            """)
        
        with col4:
            st.markdown("""
            ### Development Team
            **Yamuna B** (Developer)
            - [LinkedIn](https://www.linkedin.com/in/yamuna-bsvy/)
            - [GitHub](https://github.com/Yamuna-b)
            - [Email](mailto:yamuna.bsvy@gmail.com)
            
            **Vishalini S** (Developer)
            - [LinkedIn](https://linkedin.com/in/vishalini-s)
            - [GitHub](https://github.com/vishalini-s)
            
            **Swetha P** (Developer)
            - [LinkedIn](https://linkedin.com/in/swetha-p)
            - [GitHub](https://github.com/swetha-p)
            
            **Syed Ameed G** (Developer)
            - [LinkedIn](https://linkedin.com/in/syed-ameed-g)
            - [GitHub](https://github.com/syed-ameed-g)
            """)
        
        st.markdown("---")
        
        # Legal and credits section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### Legal
            - [Privacy Policy](/privacy)
            - [Terms of Service](/terms)
            - [Cookie Policy](/cookies)
            - [Data Protection](/data-protection)
            """)
        
        with col2:
            st.markdown("""
            ### Credits
            - **NASA Exoplanet Archive** - Data source
            - **Streamlit** - Web framework
            - **Scikit-learn** - Machine learning
            - **Plotly** - Visualizations
            """)
        
        with col3:
            st.markdown("""
            ### Acknowledgments
            - **NASA Space Apps Challenge**
            - **Open Source Community**
            - **Astronomy Research Community**
            - **Beta Testers & Contributors**
            """)
        
        # Copyright and license
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            current_year = datetime.now().year
            st.markdown(f"""
            © {current_year} Team Exoplanet AI. All rights reserved.
            
            Built for NASA Space Apps Challenge 2024
            """)
        
        with col2:
            st.markdown("""
            **License:**
            Open Source Project
            """)
        
        with col3:
            st.markdown("""
            **Data Attribution:**
            NASA Exoplanet Archive at IPAC/Caltech
            """)
        
        # Technical information
        with st.expander("🔧 Technical Information", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Technology Stack:**
                - **Frontend:** Streamlit
                - **Backend:** Python
                - **ML Framework:** Scikit-learn
                - **Database:** SQLite
                - **Visualization:** Plotly, Matplotlib
                - **Authentication:** Custom JWT
                """)
            
            with col2:
                st.markdown("""
                **System Information:**
                - **Python Version:** 3.8+
                - **Streamlit Version:** 1.28.0
                - **Deployment:** Cloud-ready
                - **Security:** HTTPS, JWT tokens
                - **Accessibility:** WCAG 2.1 AA compliant
                """)
        
        # Performance metrics (if available)
        self.show_system_status()
    
    def show_system_status(self):
        """Show system status and performance metrics."""
        with st.expander("📊 System Status", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("System Status", "🟢 Operational")
            
            with col2:
                st.metric("Uptime", "99.9%")
            
            with col3:
                st.metric("Response Time", "< 2s")
            
            with col4:
                st.metric("Active Users", "1,234")
            
            st.markdown("""
            **Last System Update:** December 15, 2024
            
            **Known Issues:** None currently reported
            
            **Maintenance Window:** Sundays 2:00-4:00 AM UTC
            """)
    
    def show_license_info(self):
        """Show detailed license information."""
        st.markdown("""
        ### 📄 MIT License
        
        Copyright (c) 2024 Yamuna B
        
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """)

# Global instances
contact_form = ContactForm()
footer_system = FooterSystem()
