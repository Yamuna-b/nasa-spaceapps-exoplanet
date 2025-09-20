"""
Help system with user guide, FAQ, and interactive tutorials.
"""
import streamlit as st
from typing import Dict, List, Any
import json

class HelpSystem:
    """Comprehensive help system for the application."""
    
    def __init__(self):
        self.faq_data = self._load_faq_data()
        self.user_guide_sections = self._load_user_guide()
    
    def show_help_page(self):
        """Show the main help page."""
        st.header("📚 Help & Documentation")
        
        # Help navigation
        help_tab = st.selectbox(
            "Choose a help topic:",
            ["Quick Start Guide", "User Guide", "FAQ", "Keyboard Shortcuts", "Troubleshooting", "Contact Support"],
            key="help_navigation"
        )
        
        if help_tab == "Quick Start Guide":
            self.show_quick_start()
        elif help_tab == "User Guide":
            self.show_user_guide()
        elif help_tab == "FAQ":
            self.show_faq()
        elif help_tab == "Keyboard Shortcuts":
            self.show_keyboard_shortcuts()
        elif help_tab == "Troubleshooting":
            self.show_troubleshooting()
        else:
            self.show_contact_support()
    
    def show_quick_start(self):
        """Show quick start guide."""
        st.subheader("Quick Start Guide")
        
        st.markdown("""
        Welcome to **Exoplanet AI**! Get started in just a few steps:
        
        ### 1. Create Your Account
        - Click **Sign Up** to create a new account
        - Verify your email address
        - Complete your profile with research interests
        
        ### 2. Configure Your Analysis
        - Select your preferred **Dataset** (Kepler, TESS, or K2)
        - Choose a **Machine Learning Model** (Random Forest, Logistic Regression, or SVM)
        - Adjust settings in your profile preferences
        
        ### 3. Upload Your Data
        - Go to **Home & Prediction**
        - Upload a CSV file with exoplanet candidate data
        - Or use our **Sample Data** to try the system
        
        ### 4. Analyze Results
        - Review prediction results and confidence scores
        - Explore **Model Performance** metrics
        - Use **Feature Analysis** to understand important variables
        
        ### 5. Advanced Features
        - Compare different models in **Model Comparison**
        - Fine-tune parameters in **Hyperparameter Tuning**
        - Export your results for further analysis
        """)
        
        # Interactive tutorial
        st.markdown("---")
        st.subheader("Interactive Tutorial")
        
        if st.button("Start Interactive Tutorial", type="primary"):
            self.start_interactive_tutorial()
        
        # Video tutorials (placeholder)
        st.markdown("---")
        st.subheader("Video Tutorials")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Getting Started** (5 min)\nLearn the basics of using Exoplanet AI")
            if st.button("Watch Tutorial 1"):
                st.info("Video tutorial would be embedded here")
        
        with col2:
            st.info("**Data Upload & Analysis** (8 min)\nHow to upload data and interpret results")
            if st.button("Watch Tutorial 2"):
                st.info("Video tutorial would be embedded here")
        
        with col3:
            st.info("**Advanced Features** (12 min)\nExplore model comparison and tuning")
            if st.button("Watch Tutorial 3"):
                st.info("Video tutorial would be embedded here")
    
    def show_user_guide(self):
        """Show detailed user guide."""
        st.subheader("📖 User Guide")
        
        # Guide sections
        guide_section = st.selectbox(
            "Select a section:",
            list(self.user_guide_sections.keys()),
            key="user_guide_section"
        )
        
        section_data = self.user_guide_sections[guide_section]
        
        st.markdown(f"### {section_data['title']}")
        st.markdown(section_data['content'])
        
        # Show subsections if available
        if 'subsections' in section_data:
            for subsection in section_data['subsections']:
                with st.expander(subsection['title']):
                    st.markdown(subsection['content'])
                    
                    if 'tips' in subsection:
                        st.info("💡 **Tips:**\n" + "\n".join(f"• {tip}" for tip in subsection['tips']))
    
    def show_faq(self):
        """Show frequently asked questions."""
        st.subheader("❓ Frequently Asked Questions")
        
        # Search FAQ
        search_term = st.text_input("🔍 Search FAQ", placeholder="Type keywords to search...")
        
        # Filter FAQ based on search
        filtered_faq = self.faq_data
        if search_term:
            filtered_faq = [
                faq for faq in self.faq_data 
                if search_term.lower() in faq['question'].lower() or 
                   search_term.lower() in faq['answer'].lower()
            ]
        
        # FAQ categories
        categories = list(set(faq['category'] for faq in filtered_faq))
        selected_category = st.selectbox("Filter by category:", ["All"] + categories)
        
        if selected_category != "All":
            filtered_faq = [faq for faq in filtered_faq if faq['category'] == selected_category]
        
        # Display FAQ
        if not filtered_faq:
            st.info("No FAQ items found matching your search.")
        else:
            for i, faq in enumerate(filtered_faq):
                with st.expander(f"❓ {faq['question']}", expanded=False):
                    st.markdown(faq['answer'])
                    
                    if 'related_links' in faq:
                        st.markdown("**Related:**")
                        for link in faq['related_links']:
                            st.markdown(f"• {link}")
        
        # Suggest new FAQ
        st.markdown("---")
        st.subheader("💬 Suggest a Question")
        
        with st.form("suggest_faq"):
            new_question = st.text_input("Your question:")
            context = st.text_area("Additional context (optional):")
            
            if st.form_submit_button("Submit Question"):
                if new_question:
                    self.submit_faq_suggestion(new_question, context)
                    st.success("Thank you! Your question has been submitted for review.")
                else:
                    st.error("Please enter a question.")
    
    def show_keyboard_shortcuts(self):
        """Show keyboard shortcuts."""
        st.subheader("⌨️ Keyboard Shortcuts")
        
        shortcuts = [
            {"keys": "Alt + H", "action": "Open Help", "description": "Opens this help system"},
            {"keys": "Alt + M", "action": "Focus Main Content", "description": "Moves focus to main content area"},
            {"keys": "Alt + S", "action": "Focus Sidebar", "description": "Moves focus to sidebar navigation"},
            {"keys": "Escape", "action": "Close Modals", "description": "Closes open modals and dropdowns"},
            {"keys": "Tab", "action": "Navigate Forward", "description": "Move to next interactive element"},
            {"keys": "Shift + Tab", "action": "Navigate Backward", "description": "Move to previous interactive element"},
            {"keys": "Enter", "action": "Activate", "description": "Activate focused button or link"},
            {"keys": "Space", "action": "Select", "description": "Toggle checkboxes or select options"},
        ]
        
        # Create shortcuts table
        for shortcut in shortcuts:
            col1, col2, col3 = st.columns([1, 2, 3])
            
            with col1:
                st.code(shortcut["keys"])
            
            with col2:
                st.write(f"**{shortcut['action']}**")
            
            with col3:
                st.write(shortcut["description"])
        
        st.markdown("---")
        st.info("💡 **Tip:** These shortcuts work throughout the application to improve accessibility and efficiency.")
    
    def show_troubleshooting(self):
        """Show troubleshooting guide."""
        st.subheader("🔧 Troubleshooting")
        
        troubleshooting_sections = {
            "Login Issues": {
                "problems": [
                    {
                        "issue": "Can't sign in with correct credentials",
                        "solutions": [
                            "Check if your email is verified (check your inbox)",
                            "Ensure Caps Lock is off when typing password",
                            "Try resetting your password",
                            "Clear browser cache and cookies",
                            "Try a different browser"
                        ]
                    },
                    {
                        "issue": "Didn't receive verification email",
                        "solutions": [
                            "Check your spam/junk folder",
                            "Wait a few minutes and check again",
                            "Ensure you entered the correct email address",
                            "Try signing up again with the same email",
                            "Contact support if the issue persists"
                        ]
                    }
                ]
            },
            "Data Upload Issues": {
                "problems": [
                    {
                        "issue": "CSV file won't upload",
                        "solutions": [
                            "Ensure file is in CSV format (.csv extension)",
                            "Check file size (should be under 200MB)",
                            "Verify CSV has proper column headers",
                            "Try saving the file again from your spreadsheet software",
                            "Use our sample data to test the system first"
                        ]
                    },
                    {
                        "issue": "Column mapping errors",
                        "solutions": [
                            "Review the column mapping interface carefully",
                            "Ensure your data columns match expected features",
                            "Use the 'create empty' option for missing columns",
                            "Check our data format documentation",
                            "Try with our sample data format as a template"
                        ]
                    }
                ]
            },
            "Performance Issues": {
                "problems": [
                    {
                        "issue": "Application is slow or unresponsive",
                        "solutions": [
                            "Refresh the browser page",
                            "Close other browser tabs to free memory",
                            "Try a different browser (Chrome, Firefox, Safari)",
                            "Check your internet connection",
                            "Clear browser cache and reload"
                        ]
                    },
                    {
                        "issue": "Predictions taking too long",
                        "solutions": [
                            "Reduce the size of your dataset",
                            "Try a simpler model (Logistic Regression vs Random Forest)",
                            "Ensure your data is properly formatted",
                            "Check if the server is experiencing high load",
                            "Contact support for large dataset processing"
                        ]
                    }
                ]
            }
        }
        
        for section_name, section_data in troubleshooting_sections.items():
            st.markdown(f"### {section_name}")
            
            for problem in section_data["problems"]:
                with st.expander(f"❗ {problem['issue']}", expanded=False):
                    st.markdown("**Try these solutions:**")
                    for i, solution in enumerate(problem['solutions'], 1):
                        st.markdown(f"{i}. {solution}")
        
        st.markdown("---")
        st.info("💡 **Still having issues?** Use the 'Report Issue' button in the sidebar or contact our support team.")
    
    def show_contact_support(self):
        """Show contact support information."""
        st.subheader("📞 Contact Support")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📧 Email Support
            **General Support:** support@exoplanet-ai.com
            
            **Technical Issues:** tech@exoplanet-ai.com
            
            **Feature Requests:** features@exoplanet-ai.com
            
            **Response Time:** Usually within 24 hours
            """)
        
        with col2:
            st.markdown("""
            ### 🌐 Other Resources
            **Documentation:** [docs.exoplanet-ai.com](https://docs.exoplanet-ai.com)
            
            **GitHub Issues:** [github.com/Yamuna-b/exoplanet-ai](https://github.com/Yamuna-b/exoplanet-ai)
            
            **Community Forum:** [community.exoplanet-ai.com](https://community.exoplanet-ai.com)
            
            **Status Page:** [status.exoplanet-ai.com](https://status.exoplanet-ai.com)
            """)
        
        st.markdown("---")
        st.subheader("📝 Contact Form")
        
        with st.form("contact_support"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Your Name")
                email = st.text_input("Your Email")
            
            with col2:
                subject_type = st.selectbox(
                    "Subject Type",
                    ["General Question", "Technical Issue", "Feature Request", "Bug Report", "Account Issue"]
                )
                priority = st.selectbox("Priority", ["Low", "Medium", "High"])
            
            subject = st.text_input("Subject", placeholder="Brief description of your inquiry")
            message = st.text_area("Message", placeholder="Please provide details about your inquiry...", height=150)
            
            include_info = st.checkbox("Include technical information to help with debugging", value=True)
            
            if st.form_submit_button("Send Message", type="primary"):
                if name and email and subject and message:
                    self.send_support_message({
                        'name': name,
                        'email': email,
                        'subject_type': subject_type,
                        'priority': priority,
                        'subject': subject,
                        'message': message,
                        'include_info': include_info
                    })
                    st.success("✅ Your message has been sent! We'll get back to you soon.")
                else:
                    st.error("Please fill in all required fields.")
    
    def start_interactive_tutorial(self):
        """Start interactive tutorial."""
        if 'tutorial_step' not in st.session_state:
            st.session_state.tutorial_step = 0
        
        tutorial_steps = [
            {
                "title": "Welcome to Exoplanet AI!",
                "content": "This tutorial will guide you through the main features of the application.",
                "action": "Let's start by exploring the sidebar navigation."
            },
            {
                "title": "Sidebar Navigation",
                "content": "The sidebar contains your user profile, configuration options, and page navigation.",
                "action": "Try changing the dataset or model selection."
            },
            {
                "title": "Home & Prediction",
                "content": "This is where you upload data and get predictions about exoplanet candidates.",
                "action": "Click 'Load Sample Data' to see how it works."
            },
            {
                "title": "Model Performance",
                "content": "View detailed metrics about how well the AI model is performing.",
                "action": "Check out the accuracy scores and confusion matrix."
            },
            {
                "title": "Feature Analysis",
                "content": "Understand which features are most important for exoplanet detection.",
                "action": "Look at the feature importance chart."
            }
        ]
        
        current_step = st.session_state.tutorial_step
        
        if current_step < len(tutorial_steps):
            step = tutorial_steps[current_step]
            
            st.info(f"**Step {current_step + 1}/{len(tutorial_steps)}: {step['title']}**")
            st.write(step['content'])
            st.write(f"👉 {step['action']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Previous", disabled=current_step == 0):
                    st.session_state.tutorial_step -= 1
                    st.rerun()
            
            with col2:
                if st.button("Next", disabled=current_step == len(tutorial_steps) - 1):
                    st.session_state.tutorial_step += 1
                    st.rerun()
            
            with col3:
                if st.button("Skip Tutorial"):
                    del st.session_state.tutorial_step
                    st.rerun()
        else:
            st.success("🎉 Tutorial completed! You're ready to explore Exoplanet AI.")
            if st.button("Restart Tutorial"):
                st.session_state.tutorial_step = 0
                st.rerun()
    
    def _load_faq_data(self) -> List[Dict[str, Any]]:
        """Load FAQ data."""
        return [
            {
                "category": "Getting Started",
                "question": "What is Exoplanet AI?",
                "answer": "Exoplanet AI is a machine learning platform that helps identify exoplanet candidates from astronomical data. It uses NASA datasets from missions like Kepler, TESS, and K2 to train AI models that can automatically classify potential exoplanets."
            },
            {
                "category": "Getting Started",
                "question": "Do I need to create an account?",
                "answer": "Yes, you need to create a free account to use Exoplanet AI. This allows us to save your preferences, analysis history, and provide personalized features."
            },
            {
                "category": "Data Upload",
                "question": "What data format should I use?",
                "answer": "Upload your data as a CSV file with columns matching the expected features for your chosen dataset (Kepler, TESS, or K2). The system will help you map your columns to the required features."
            },
            {
                "category": "Data Upload",
                "question": "What's the maximum file size I can upload?",
                "answer": "Currently, you can upload CSV files up to 200MB in size. For larger datasets, please contact our support team."
            },
            {
                "category": "Models",
                "question": "Which machine learning model should I choose?",
                "answer": "**Random Forest** is generally the most accurate but slower. **Logistic Regression** is faster and more interpretable. **SVM** works well for smaller datasets. Try different models and compare their performance."
            },
            {
                "category": "Models",
                "question": "How accurate are the predictions?",
                "answer": "Model accuracy varies by dataset and model type, typically ranging from 85-95%. Check the Model Performance page for detailed metrics including precision, recall, and cross-validation scores."
            },
            {
                "category": "Results",
                "question": "What does the confidence score mean?",
                "answer": "The confidence score (0-100%) indicates how certain the model is about its prediction. Higher scores mean the model is more confident. We recommend focusing on predictions with >80% confidence."
            },
            {
                "category": "Results",
                "question": "Can I export my results?",
                "answer": "Yes! You can download your prediction results as CSV files. The export includes all original data plus predictions and confidence scores."
            },
            {
                "category": "Technical",
                "question": "What datasets are supported?",
                "answer": "We support three NASA mission datasets: **Kepler** (original mission), **TESS** (Transiting Exoplanet Survey Satellite), and **K2** (extended Kepler mission). Each has different features and characteristics."
            },
            {
                "category": "Technical",
                "question": "How do I interpret the feature importance?",
                "answer": "Feature importance shows which measurements are most useful for detecting exoplanets. Higher values mean the feature has more influence on the model's decisions. This helps understand what makes a good exoplanet candidate."
            },
            {
                "category": "Account",
                "question": "How do I reset my password?",
                "answer": "Click 'Forgot Password' on the sign-in page, enter your email address, and follow the instructions in the reset email. Check your spam folder if you don't see the email."
            },
            {
                "category": "Account",
                "question": "Can I connect my Google or GitHub account?",
                "answer": "Yes! You can sign in with Google or GitHub OAuth, or connect these accounts to your existing profile in the Account Settings."
            }
        ]
    
    def _load_user_guide(self) -> Dict[str, Dict[str, Any]]:
        """Load user guide sections."""
        return {
            "Authentication": {
                "title": "User Authentication & Accounts",
                "content": """
                Learn how to create and manage your Exoplanet AI account.
                """,
                "subsections": [
                    {
                        "title": "Creating an Account",
                        "content": """
                        1. Click **Sign Up** on the main page
                        2. Enter your full name, email, and a secure password
                        3. Accept the terms of service
                        4. Check your email for a verification link
                        5. Click the verification link to activate your account
                        """,
                        "tips": [
                            "Use a strong password with special characters",
                            "Check your spam folder for the verification email",
                            "You can also sign up with Google or GitHub"
                        ]
                    },
                    {
                        "title": "Profile Management",
                        "content": """
                        Customize your profile to get the most out of Exoplanet AI:
                        
                        - **Basic Information**: Name, bio, institution
                        - **Research Interests**: Your areas of focus
                        - **Privacy Settings**: Control who can see your profile
                        - **Preferences**: Default datasets, models, and UI settings
                        """,
                        "tips": [
                            "Complete your profile to connect with other researchers",
                            "Set your default preferences to save time",
                            "Enable collaboration invites to work with others"
                        ]
                    }
                ]
            },
            "Data Analysis": {
                "title": "Data Upload & Analysis",
                "content": """
                Learn how to upload your data and get exoplanet predictions.
                """,
                "subsections": [
                    {
                        "title": "Preparing Your Data",
                        "content": """
                        Your CSV file should contain the following types of features:
                        
                        **For Kepler data:**
                        - `koi_period`: Orbital period (days)
                        - `koi_prad`: Planet radius (Earth radii)
                        - `koi_depth`: Transit depth (ppm)
                        - `koi_duration`: Transit duration (hours)
                        
                        **For TESS data:**
                        - `pl_orbper`: Orbital period (days)
                        - `pl_rade`: Planet radius (Earth radii)
                        - `tran_depth`: Transit depth (ppm)
                        - `tran_dur14`: Transit duration (hours)
                        """,
                        "tips": [
                            "Column names don't need to match exactly - use the mapping interface",
                            "Missing values will be handled automatically",
                            "Try our sample data first to understand the format"
                        ]
                    },
                    {
                        "title": "Understanding Results",
                        "content": """
                        After analysis, you'll see:
                        
                        - **Predictions**: Classification for each candidate
                        - **Confidence Scores**: How certain the model is (0-100%)
                        - **Summary Statistics**: Count of each prediction type
                        - **Visualizations**: Charts showing confidence distribution
                        """,
                        "tips": [
                            "Focus on high-confidence predictions (>80%)",
                            "Use the detailed results table for analysis",
                            "Export results for further processing"
                        ]
                    }
                ]
            }
        }
    
    def submit_faq_suggestion(self, question: str, context: str):
        """Submit a new FAQ suggestion."""
        # In a real application, this would save to a database
        suggestion = {
            'question': question,
            'context': context,
            'timestamp': st.session_state.get('current_time', 'unknown'),
            'user_id': st.session_state.get('user', {}).get('user_id', 'anonymous')
        }
        
        # Log the suggestion
        print(f"FAQ Suggestion: {suggestion}")
    
    def send_support_message(self, message_data: Dict[str, Any]):
        """Send support message."""
        # In a real application, this would send an email or create a support ticket
        print(f"Support Message: {message_data}")

# Global help system instance
help_system = HelpSystem()
